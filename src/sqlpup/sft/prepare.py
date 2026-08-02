"""Stream raw SFT rows into filtered, tokenized pair files (torch-free).

SynSQL's ``data.json`` is one multi-gigabyte JSON array; :func:`iter_json_array`
walks it with a bounded buffer and a real JSON decoder per element (never a
regex, never a whole-file load), so preparation runs in constant memory on a
small box. Pairs are built through the same seam-verified
:func:`~sqlpup.sft.pairs.build_pair` the tests pin; anything that cannot be
built *safely* is counted, never silently included: missing databases are
skipped, over-context pairs are filtered (truncating DDL mid-table would teach
malformed schemas), and boundary violations are tallied loudly.

Sampling is per-row deterministic (seeded hash), so a 40%% sample is
reproducible from ``(seed, sample_rate)`` alone and two runs ship identical
bytes -- the manifest records everything a paper needs to restate the dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Final

from sqlpup.eval.prompts import BIRD_DDL_V1, PromptSpec, schema_ddl
from sqlpup.sft.linking import linked_target, schema_identifiers
from sqlpup.sft.pairs import DEFAULT_CONTEXT_LIMIT, BoundaryError, TokenizerLike, build_pair
from sqlpup.sft.truncate import reduced_ddl

_DEFAULT_BUFFER: Final = 1 << 20  # 1 MiB read granularity
# Rows per IPC hand-off when parallel: big enough to amortise pickling, small
# enough that in-flight rows stay a bounded slice of memory.
_CHUNKSIZE: Final = 64


def iter_json_array(path: Path, *, buffer_bytes: int = _DEFAULT_BUFFER) -> Iterator[dict[str, Any]]:
    """Yield objects from a JSON array file without loading the whole file.

    Scans for element boundaries with a quote/escape-aware brace counter, then
    hands each candidate slice to ``json.loads`` -- the real decoder still
    validates every element, the scanner only finds the cut points.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buf = ""
        # consume the leading '['
        while "[" not in buf:
            chunk = handle.read(buffer_bytes)
            if not chunk:
                return
            buf += chunk
        buf = buf[buf.index("[") + 1 :]
        depth = 0
        in_string = False
        escaped = False
        start: int | None = None
        pos = 0
        while True:
            if pos >= len(buf):
                chunk = handle.read(buffer_bytes)
                if not chunk:
                    return
                buf += chunk
            char = buf[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = pos
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    obj, _ = decoder.raw_decode(buf[start : pos + 1])
                    yield obj
                    buf = buf[pos + 1 :]
                    pos = -1
                    start = None
            pos += 1


def _normalize(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """(db_id, question, evidence, sql) across the three source dialects.

    SynSQL says ``sql``/``external_knowledge``, BIRD-train says ``SQL``/
    ``evidence``, Spider says ``query`` and has no evidence field -- one
    mixed-run reader beats three prepare variants.
    """
    sql = row.get("sql") or row.get("SQL") or row.get("query")
    if not sql:
        raise KeyError(f"row has no sql/SQL/query field: {sorted(row)}")
    evidence = row.get("external_knowledge") or row.get("evidence") or ""
    return str(row["db_id"]), str(row.get("question", "")), str(evidence), str(sql)


# --- per-row pair building, shared by the serial and parallel paths ----------
# One implementation, two drivers: workers change throughput, never output.
# Worker state lives in a module global because multiprocessing initialises
# each process once and reuses it (the tokenizer and the per-database caches
# are far too expensive to rebuild per row).
_WORKER: dict[str, Any] = {}


def _setup_worker(
    tokenizer: TokenizerLike,
    db_root: Path,
    spec: PromptSpec,
    eos_id: int,
    context_limit: int,
    link_targets: bool,
    reduce_overflow: bool,
) -> None:
    _WORKER.clear()
    _WORKER.update(
        tokenizer=tokenizer,
        db_root=Path(db_root),
        spec=spec,
        eos_id=eos_id,
        context_limit=context_limit,
        link_targets=link_targets,
        reduce_overflow=reduce_overflow,
        ddl_cache={},
        schema_cache={},
    )


def _init_worker(
    tokenizer_path: str,
    db_root: Path,
    spec: PromptSpec,
    eos_id: int,
    context_limit: int,
    link_targets: bool,
    reduce_overflow: bool,
) -> None:  # pragma: no cover - runs in a child process
    from tokenizers import Tokenizer

    _setup_worker(
        Tokenizer.from_file(tokenizer_path),
        db_root,
        spec,
        eos_id,
        context_limit,
        link_targets,
        reduce_overflow,
    )


def _build_one(row: Mapping[str, Any]) -> tuple[str | None, str]:
    """Build one row's pair. Returns ``(json_line_or_None, outcome_tag)``."""
    w = _WORKER
    db_id, question, evidence, sql = _normalize(row)
    ddl_cache: dict[str, str | None] = w["ddl_cache"]
    db_path = w["db_root"] / db_id / f"{db_id}.sqlite"
    if db_id not in ddl_cache:
        ddl_cache[db_id] = schema_ddl(db_path) if db_path.exists() else None
    ddl = ddl_cache[db_id]
    if ddl is None:
        return None, "missing_db"

    completion = sql
    if w["link_targets"]:
        schema_cache: dict[str, dict[str, list[str]]] = w["schema_cache"]
        if db_id not in schema_cache:
            schema_cache[db_id] = schema_identifiers(db_path)
        completion = linked_target(sql, schema_cache[db_id]) + sql

    tag = "ok"
    try:
        pair = build_pair(
            w["tokenizer"],
            ddl=ddl,
            question=question,
            evidence=evidence,
            sql=completion,
            eos_id=w["eos_id"],
            spec=w["spec"],
            context_limit=w["context_limit"],
        )
        if pair.overflow and w["reduce_overflow"]:
            for level in (1, 2):
                pair = build_pair(
                    w["tokenizer"],
                    ddl=reduced_ddl(db_path, sql, level=level),
                    question=question,
                    evidence=evidence,
                    sql=completion,
                    eos_id=w["eos_id"],
                    spec=w["spec"],
                    context_limit=w["context_limit"],
                )
                if not pair.overflow:
                    tag = f"ok_reduced{level}"
                    break
    except BoundaryError:
        return None, "boundary"
    if pair.overflow:
        return None, "overflow"
    line = json.dumps(
        {
            "input_ids": list(pair.input_ids),
            "labels": list(pair.labels),
            "prompt_tokens": pair.prompt_tokens,
        }
    )
    return line, tag


def _keep(index: int, sample_rate: float, seed: int) -> bool:
    """Deterministic per-row keep decision (stable across runs and machines)."""
    if sample_rate >= 1.0:
        return True
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < sample_rate


def prepare_pairs(
    rows: Iterable[Mapping[str, Any]],
    db_root: Path,
    tokenizer: TokenizerLike,
    out_path: Path,
    *,
    eos_id: int,
    spec: PromptSpec = BIRD_DDL_V1,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    sample_rate: float = 1.0,
    seed: int = 0,
    limit: int | None = None,
    link_targets: bool = False,
    reduce_overflow: bool = False,
    workers: int = 1,
    tokenizer_path: str | None = None,
) -> dict[str, Any]:
    """Build pairs for every kept row; write JSONL + return the manifest.

    ``link_targets`` prefixes every completion with the schema-linking block
    derived from the gold SQL (v3 targets). ``reduce_overflow`` retries an
    over-context pair with level-1 then level-2 reduced DDL before giving up
    -- recovering the large-schema pairs v2e discarded outright.

    ``workers`` > 1 spreads the per-row work (tokenisation dominates, and the
    reduction ladder re-tokenises) across processes; rows are dispatched and
    written *in order*, so the output file is byte-identical to a serial run
    (pinned by test). Each worker rebuilds the tokenizer from
    ``tokenizer_path``, which is therefore required.
    """
    if workers > 1 and tokenizer_path is None:
        raise ValueError("workers > 1 requires tokenizer_path (each process rebuilds it)")

    counts = {"ok": 0, "ok_reduced1": 0, "ok_reduced2": 0}
    skipped_missing_db = 0
    filtered_overflow = 0
    boundary_errors = 0
    seen = 0

    def _kept_rows() -> Iterator[Mapping[str, Any]]:
        nonlocal seen
        for index, row in enumerate(rows):
            if not _keep(index, sample_rate, seed):
                continue
            seen += 1
            yield row

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        if workers > 1:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            init_args = (
                tokenizer_path,
                db_root,
                spec,
                eos_id,
                context_limit,
                link_targets,
                reduce_overflow,
            )
            pool = ctx.Pool(workers, initializer=_init_worker, initargs=init_args)
            try:
                results: Iterator[tuple[str | None, str]] = pool.imap(
                    _build_one, _kept_rows(), chunksize=_CHUNKSIZE
                )
                for line, tag in results:
                    if line is None:
                        if tag == "missing_db":
                            skipped_missing_db += 1
                        elif tag == "overflow":
                            filtered_overflow += 1
                        else:
                            boundary_errors += 1
                        continue
                    out.write(line + "\n")
                    counts[tag] += 1
                    if limit is not None and sum(counts.values()) >= limit:
                        break
            finally:
                pool.terminate()
                pool.join()
        else:
            _setup_worker(
                tokenizer, db_root, spec, eos_id, context_limit, link_targets, reduce_overflow
            )
            for row in _kept_rows():
                line, tag = _build_one(row)
                if line is None:
                    if tag == "missing_db":
                        skipped_missing_db += 1
                    elif tag == "overflow":
                        filtered_overflow += 1
                    else:
                        boundary_errors += 1
                    continue
                out.write(line + "\n")
                counts[tag] += 1
                if limit is not None and sum(counts.values()) >= limit:
                    break

    return {
        "written": sum(counts.values()),
        "seen": seen,
        "skipped_missing_db": skipped_missing_db,
        "filtered_overflow": filtered_overflow,
        "boundary_errors": boundary_errors,
        "reduced_level1": counts["ok_reduced1"],
        "reduced_level2": counts["ok_reduced2"],
        "targets": "linked-v1" if link_targets else "plain",
        "prompt_spec": spec.spec_id,
        "context_limit": context_limit,
        "sample_rate": sample_rate,
        "seed": seed,
        "workers": workers,
        "out": str(out_path),
    }
