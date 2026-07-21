"""Streaming corpus download with token budgets and resume manifests.

Sources stream record-by-record (never fully materialized), stop at their
configured token budget (approximated as characters x 0.25 until the real
tokenizer exists), and write a manifest next to the output so an interrupted
or repeated run resumes instead of re-downloading.

Network access is confined to the fetch readers (:func:`hf_stream_reader`,
:func:`schemapile_reader`, :func:`synsql_reader`) and :func:`_download_artifact`;
every other code path takes an injected record iterator or a pre-fetched cache
file, which is what the offline tests use.
"""

from __future__ import annotations

import codecs
import gzip
import json
import os
import shutil
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from sqlpup.config import SourceConfig
from sqlpup.data.render import RenderError, render_schemapile, render_synsql
from sqlpup.io import Document, write_documents

APPROX_TOKENS_PER_CHAR = 0.25

RecordReader = Callable[[SourceConfig], Iterator[Mapping[str, Any]]]


class MissingTextError(ValueError):
    """Raised when a record lacks usable text for its configured field."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    source: str
    docs: int
    approx_tokens: int
    skipped: int
    path: Path
    resumed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "docs": self.docs,
            "approx_tokens": self.approx_tokens,
            "skipped": self.skipped,
            "path": str(self.path),
            "resumed": self.resumed,
        }


def hf_stream_reader(cfg: SourceConfig) -> Iterator[Mapping[str, Any]]:
    """Stream records from the Hugging Face Hub without downloading full shards.

    Most subsets are builder configs selected with ``name=`` (e.g. FineWeb-Edu's
    ``sample-10BT``). Directory-style subsets, ``bigcode/starcoderdata``'s
    ``sql`` and ``bigcode/the-stack-dedup``'s ``data/python``, are folders in
    the dataset repo and must be selected with ``data_dir=`` instead; passing
    them as ``name=`` raises. Those sources set ``hf_data_dir`` (leaving
    ``hf_name`` null), so the two paths are mutually exclusive here.
    """
    import datasets

    if cfg.hf_path is None:
        raise ValueError(f"source {cfg.name!r} has no hf_path; provide a custom reader")
    if cfg.hf_data_dir is not None:
        dataset = datasets.load_dataset(
            cfg.hf_path, data_dir=cfg.hf_data_dir, split=cfg.split, streaming=True
        )
    else:
        dataset = datasets.load_dataset(
            cfg.hf_path, name=cfg.hf_name, split=cfg.split, streaming=True
        )
    for record in dataset:
        if isinstance(record, Mapping):
            yield record


_FETCH_USER_AGENT = "sqlpup-data-fetcher"
_HTTP_TIMEOUT = 30  # seconds; a dead server errors instead of hanging forever
_CHUNK_SIZE = 1 << 20  # 1 MiB streamed read granularity


def _download_artifact(url: str, dest: Path) -> None:
    """Stream an HTTP artifact to ``dest`` in chunks, never holding it in memory.

    Writes to a ``.part`` sibling and renames on success so an interrupted run
    leaves no half-written cache file to be mistaken for a complete download.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _FETCH_USER_AGENT})
    with (
        urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response,
        tmp.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=_CHUNK_SIZE)
    tmp.replace(dest)


def _iter_read_chunks(source: IO[bytes], chunk_size: int) -> Iterator[bytes]:
    """Yield fixed-size blocks from any binary ``read``-able (socket or file)."""
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            return
        yield chunk


def _iter_http_chunks(url: str, chunk_size: int = _CHUNK_SIZE) -> Iterator[bytes]:
    """Stream an HTTP body in chunks; the socket closes when iteration stops.

    Because this is a generator, a consumer that stops early (e.g. at a doc cap)
    closes it, which exits the ``urlopen`` context and drops the connection,
    so a capped run reads only a prefix of a huge file instead of all of it.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _FETCH_USER_AGENT})
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
        yield from _iter_read_chunks(response, chunk_size)


def _iter_json_array(chunks: Iterable[bytes]) -> Iterator[Any]:
    """Yield each top-level value of a JSON array from streamed byte chunks.

    Values are peeled off one at a time with :meth:`json.JSONDecoder.raw_decode`,
    so a multi-gigabyte array is parsed object-by-object without ever being held
    whole: only the current value (plus at most one unconsumed chunk) is
    buffered, and the caller can stop early. An incremental UTF-8 decoder keeps
    multibyte characters intact when they straddle a chunk boundary. Only the
    array framing (leading ``[``, inter-value commas, whitespace) is handled
    here; each value itself is decoded by the stdlib JSON parser.
    """
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()
    stream = iter(chunks)
    buffer = ""
    pos = 0
    started = False

    def pull() -> bool:
        """Drop the consumed prefix and append the next chunk; False at EOF."""
        nonlocal buffer, pos
        if pos:
            buffer = buffer[pos:]
            pos = 0
        try:
            chunk = next(stream)
        except StopIteration:
            buffer += utf8.decode(b"", final=True)
            return False
        buffer += utf8.decode(chunk)
        return True

    while True:
        while pos < len(buffer) and buffer[pos] in " \t\r\n,":
            pos += 1
        if not started:
            if pos < len(buffer):
                if buffer[pos] != "[":
                    raise ValueError("expected a JSON array")
                pos += 1
                started = True
                continue
            if not pull():
                return
            continue
        if pos < len(buffer) and buffer[pos] == "]":
            return
        if pos >= len(buffer):
            if not pull():
                # Natural exhaustion after the opening '[' but before the
                # closing ']' -- a truncated stream (e.g. a cut-off 9.36GB
                # download), not a clean end. Only reached once ``started``, so
                # it always means "opened, never closed". A capped consumer
                # instead tears this generator down at the ``yield`` below
                # (GeneratorExit), which never re-enters this loop, so the guard
                # cannot fire on an intentional early stop.
                raise ValueError("truncated JSON array: stream ended without closing ']'")
            continue
        try:
            value, end = decoder.raw_decode(buffer, pos)
        except json.JSONDecodeError:
            if not pull():
                raise
            continue
        pos = end
        yield value


def _iter_schemapile(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield one record per schema from a SchemaPile gzipped-JSON artifact."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a JSON object keyed by schema name")
    for name, value in data.items():
        if isinstance(value, Mapping):
            yield {"schema_name": name, "INFO": value.get("INFO"), "TABLES": value.get("TABLES")}


def schemapile_reader(cfg: SourceConfig, out_dir: Path) -> Iterator[Mapping[str, Any]]:
    """Fetch (once, cached) and stream the SchemaPile artifact named by ``cfg.url``.

    The gzipped JSON is a single object, so it is cached under ``out_dir`` and
    decoded once; the token budget in :func:`download_source` bounds how many
    of its schemas are actually rendered.
    """
    if cfg.url is None:
        raise ValueError(f"source {cfg.name!r} has kind 'schemapile' but no url")
    cache = out_dir / f"{cfg.name}.source.json.gz"
    if not cache.exists():
        _download_artifact(cfg.url, cache)
    yield from _iter_schemapile(cache)


def _load_ddl_index(path: Path) -> dict[str, list[str]]:
    """Map each ``db_id`` to its list of ``CREATE TABLE`` DDL statements.

    Built by streaming the (cached) SynSQL ``tables.json`` array one record at a
    time, so the ~306MB file is never loaded whole. Records without a ``db_id``
    or with no usable DDL are skipped, so a data.json record joining to them is
    treated as missing (and skipped by the renderer).
    """
    index: dict[str, list[str]] = {}
    with path.open("rb") as handle:
        for record in _iter_json_array(_iter_read_chunks(handle, _CHUNK_SIZE)):
            if not isinstance(record, Mapping):
                continue
            db_id = record.get("db_id")
            ddls = record.get("ddls")
            if not isinstance(db_id, str) or not isinstance(ddls, list):
                continue
            statements = [d for d in ddls if isinstance(d, str) and d.strip()]
            if statements:
                index[db_id] = statements
    return index


def _join_synsql(
    records: Iterable[Any], index: Mapping[str, Sequence[str]]
) -> Iterator[Mapping[str, Any]]:
    """Attach each record's schema ``ddls`` from ``index`` by ``db_id``.

    A record whose ``db_id`` is absent from the index is yielded unchanged (no
    ``ddls``), so :func:`render_synsql` raises and the caller counts it skipped.
    """
    for record in records:
        if not isinstance(record, Mapping):
            continue
        merged = dict(record)
        db_id = record.get("db_id")
        if isinstance(db_id, str):
            ddls = index.get(db_id)
            if ddls is not None:
                merged["ddls"] = list(ddls)
        yield merged


def synsql_reader(cfg: SourceConfig, out_dir: Path) -> Iterator[Mapping[str, Any]]:
    """Stream SynSQL's data.json, joining each record's schema DDL by ``db_id``.

    The ~9.36GB ``data.json`` array is parsed object-by-object straight off the
    socket (see :func:`_iter_json_array`) and never fully downloaded: once
    :func:`download_source` stops at its doc cap or token budget, this generator
    is closed and the connection drops. The much smaller ``tables.json`` is
    cached once under ``out_dir`` and turned into an in-memory ``db_id -> ddls``
    index (its ~306MB fits comfortably in memory).
    """
    if cfg.url is None:
        raise ValueError(f"source {cfg.name!r} has kind 'synsql' but no url")
    if cfg.tables_url is None:
        raise ValueError(f"source {cfg.name!r} has kind 'synsql' but no tables_url")
    tables_cache = out_dir / f"{cfg.name}.tables.json"
    if not tables_cache.exists():
        _download_artifact(cfg.tables_url, tables_cache)
    index = _load_ddl_index(tables_cache)
    yield from _join_synsql(_iter_json_array(_iter_http_chunks(cfg.url)), index)


def _select_reader(cfg: SourceConfig, out_dir: Path) -> RecordReader:
    """Pick the record reader for a source's fetch ``kind``."""
    if cfg.kind == "schemapile":

        def read_schemapile(source: SourceConfig) -> Iterator[Mapping[str, Any]]:
            return schemapile_reader(source, out_dir)

        return read_schemapile
    if cfg.kind == "synsql":

        def read_synsql(source: SourceConfig) -> Iterator[Mapping[str, Any]]:
            return synsql_reader(source, out_dir)

        return read_synsql
    return hf_stream_reader


def record_to_document(cfg: SourceConfig, record: Mapping[str, Any], index: int) -> Document:
    """Turn a raw record into a Document, rendering structured sources."""
    doc_id = f"{cfg.name}-{index}"
    if cfg.text_field is not None:
        value = record.get(cfg.text_field)
        if not isinstance(value, str) or not value.strip():
            raise MissingTextError(f"{doc_id}: empty or missing field {cfg.text_field!r}")
        return Document(id=doc_id, text=value, source=cfg.name)
    # Dispatch on the fetch ``kind`` (what ``_select_reader`` keys on), not the
    # source ``name`` -- so a second synsql/schemapile source under a different
    # name (anneal slice, synth schema-Q&A, hard slice) still finds its renderer.
    if cfg.kind == "synsql":
        return render_synsql(record, doc_id)
    if cfg.kind == "schemapile":
        return render_schemapile(record, doc_id)
    raise ValueError(f"source {cfg.name!r} has no text_field and no registered renderer")


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Write the resume manifest and fsync it to disk before returning.

    The download CLI exits abruptly via ``os._exit`` on success (to skip a
    libarrow teardown that can otherwise hang the process); fsyncing here means
    that fast exit can never leave a truncated or missing manifest that would
    corrupt a later resume.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())


def download_source(
    cfg: SourceConfig, out_dir: Path, reader: RecordReader | None = None
) -> DownloadResult:
    """Stream one source to ``<out_dir>/<name>.jsonl.zst``, respecting budgets."""
    if reader is None:
        reader = _select_reader(cfg, out_dir)
    out_path = out_dir / f"{cfg.name}.jsonl.zst"
    manifest_path = out_dir / f"{cfg.name}.manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete"):
            return DownloadResult(
                source=cfg.name,
                docs=int(manifest["docs"]),
                approx_tokens=int(manifest["approx_tokens"]),
                skipped=int(manifest.get("skipped", 0)),
                path=out_path,
                resumed=True,
            )

    char_budget = int(cfg.target_tokens / APPROX_TOKENS_PER_CHAR)
    docs = 0
    chars = 0
    skipped = 0

    def generate() -> Iterator[Document]:
        nonlocal docs, chars, skipped
        for index, record in enumerate(reader(cfg)):
            if chars >= char_budget:
                break
            if cfg.doc_cap is not None and docs >= cfg.doc_cap:
                break
            try:
                doc = record_to_document(cfg, record, index)
            except (MissingTextError, RenderError):
                skipped += 1
                continue
            docs += 1
            chars += len(doc.text)
            yield doc

    write_documents(out_path, generate())
    result = DownloadResult(
        source=cfg.name,
        docs=docs,
        approx_tokens=int(chars * APPROX_TOKENS_PER_CHAR),
        skipped=skipped,
        path=out_path,
    )
    _write_manifest(manifest_path, {**result.as_dict(), "complete": True})
    return result
