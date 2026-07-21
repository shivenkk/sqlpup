"""Fetch evaluation sets and build the decontamination n-gram index.

BIRD dev and Spider dev/test ship as zip archives bundling their databases;
only the JSON member named in the eval-sets config is extracted (stdlib
``zipfile``): the surrounding hundreds of megabytes are never unpacked. Each
configured field of each example becomes its own index text entry (the
question, the gold SQL/query, and BIRD's ``evidence`` where non-empty), which
is the design spec's "questions + SQL" decontamination binding with evidence
included as a strictly-safer superset. The built :class:`NgramIndex` is
persisted next to a JSON provenance report (per-source example and text-entry
counts, each source's URL, size, and sha256, the n-gram width, and a UTC
timestamp) so a published decontamination run is exactly reproducible.

Network access is confined to :func:`fetch_archive`, which reuses
``download._download_artifact`` (cached, atomic, timed out). Extraction, index
building, and report writing all operate on a local cached path, which is what
the offline tests exercise.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlpup.config import EvalSetsConfig, EvalSourceConfig
from sqlpup.data.decontaminate import DEFAULT_NGRAM, NgramIndex

_HASH_CHUNK = 1 << 20  # 1 MiB streamed read granularity


def _cache_path(out_dir: Path, source: EvalSourceConfig) -> Path:
    """Stable per-URL cache path so sources sharing an archive download once."""
    digest = hashlib.sha256(source.url.encode("utf-8")).hexdigest()[:16]
    suffix = ".zip" if source.archive_member else ".json"
    return out_dir / f"{digest}{suffix}"


def fetch_archive(source: EvalSourceConfig, out_dir: Path) -> Path:
    """Download the source artifact once (cached) and return its local path."""
    from sqlpup.data import download

    cache = _cache_path(out_dir, source)
    if not cache.exists():
        download._download_artifact(source.url, cache)
    return cache


def _sha256_and_size(path: Path) -> tuple[str, int]:
    """Content sha256 and byte size of ``path``, read in chunks (never whole)."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def read_member(path: Path, member: str | None) -> bytes:
    """Return the bytes of ``member`` inside the zip at ``path``.

    With ``member`` None the whole file is read (a source distributed as a bare
    JSON file rather than a zip). A configured member that is absent raises
    :class:`KeyError` naming it, so a wrong path fails loudly instead of
    silently indexing nothing.
    """
    if member is None:
        return path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        if member not in set(archive.namelist()):
            raise KeyError(f"{path}: member {member!r} not found in archive")
        return archive.read(member)


def _load_examples(raw: bytes, source_name: str) -> list[Mapping[str, Any]]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"{source_name}: expected a JSON list of examples")
    examples: list[Mapping[str, Any]] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError(f"{source_name}: expected objects in the list")
        examples.append(item)
    return examples


def extract_field_texts(examples: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> list[str]:
    """Emit each configured field of each example as its own index text entry.

    A field contributes only when present and non-empty after stripping, so
    BIRD's frequently-blank ``evidence`` adds nothing when it is absent or empty.
    """
    texts: list[str] = []
    for example in examples:
        for field in fields:
            value = example.get(field)
            if isinstance(value, str) and value.strip():
                texts.append(value)
    return texts


def build_eval_index(
    config: EvalSetsConfig,
    out_dir: Path,
    index_path: Path,
    report_path: Path,
    n: int = DEFAULT_NGRAM,
) -> dict[str, Any]:
    """Fetch every eval set, build the n-gram index, and write it plus a report.

    Returns the report dict (also written to ``report_path``). The index is
    deterministic given identical inputs; only the report's ``generated_at``
    timestamp varies between runs.
    """
    provenance: dict[Path, tuple[str, int]] = {}
    all_texts: list[str] = []
    sources: list[dict[str, Any]] = []

    for source in config.sources:
        archive = fetch_archive(source, out_dir)
        if archive not in provenance:
            provenance[archive] = _sha256_and_size(archive)
        sha256, size_bytes = provenance[archive]
        examples = _load_examples(read_member(archive, source.archive_member), source.name)
        if source.expected_examples is not None and len(examples) != source.expected_examples:
            raise ValueError(
                f"{source.name}: expected {source.expected_examples} examples but got "
                f"{len(examples)} from {source.url} (member {source.archive_member!r}); "
                "the distribution may have changed or the download is incomplete"
            )
        texts = extract_field_texts(examples, source.fields)
        all_texts.extend(texts)
        sources.append(
            {
                "name": source.name,
                "url": source.url,
                "member": source.archive_member,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "examples": len(examples),
                "expected_examples": source.expected_examples,
                "text_entries": len(texts),
            }
        )

    index = NgramIndex.build(all_texts, n=n)
    index.save(index_path)

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n": index.n,
        "total_shingles": len(index.grams),
        "sources": sources,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
