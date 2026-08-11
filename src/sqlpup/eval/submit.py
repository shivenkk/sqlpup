"""Predict on a held-out BIRD split and write the file their evaluator reads.

Three things separate a held-out run from a dev run, and each is handled here
rather than by asking the caller to remember it.

*Gold is not available.* Examples arrive through
:func:`~sqlpup.eval.dataset.load_prediction_examples`, whose gold field is always
``""``. Nothing downstream reads it, so a held-out run cannot start depending on
an answer it does not have.

*A crash must not cost the whole run.* Generation is chunked, and every finished
chunk is appended to a JSONL progress file before the next one starts. Re-running
the same command skips what is already there. BIRD asks for this explicitly so a
failed evaluation resumes from the failing example instead of from the top.

*The output format is theirs, not ours.* Their evaluator reads
``predict_<split>.json`` as a mapping from stringified position to
``"<sql>\\t----- bird -----\\t<db_id>"`` and splits on that separator, so the SQL
is flattened to one line: an embedded tab would silently split into the wrong
fields, and a newline would survive JSON but not the shell pipelines around it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Final

from sqlpup.eval.dataset import BirdExample

BIRD_SEPARATOR: Final = "\t----- bird -----\t"

logger = logging.getLogger(__name__)


def bird_prediction(sql: str, db_id: str) -> str:
    """One entry of a BIRD prediction file, flattened to a single line."""
    return f"{' '.join(sql.split())}{BIRD_SEPARATOR}{db_id}"


def write_bird_predictions(
    records: Sequence[dict[str, Any]], examples: Sequence[BirdExample], path: Path | str
) -> int:
    """Write ``predict_<split>.json`` keyed by position, in example order.

    Their evaluator zips this file's values against a gold list by iteration
    order, so a missing example would silently shift every answer after it onto
    the wrong question. Any gap is an error here instead.
    """
    by_index = {int(record["index"]): record for record in records}
    missing = [example.index for example in examples if example.index not in by_index]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(examples)} examples have no prediction "
            f"(first: index {missing[0]}); refusing to write a misaligned file"
        )
    payload = {
        str(example.index): bird_prediction(
            by_index[example.index].get("predicted_sql", ""), example.db_id
        )
        for example in examples
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sum(1 for value in payload.values() if not value.split(BIRD_SEPARATOR)[0])


def _read_progress(path: Path) -> dict[int, dict[str, Any]]:
    """Records from a previous run, keyed by index. Truncated lines are dropped.

    A run killed mid-write leaves a partial final line; that example is simply
    regenerated, which is cheaper than refusing to resume.
    """
    if not path.exists():
        return {}
    done: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("dropping a truncated progress line; that example is regenerated")
            continue
        done[int(record["index"])] = record
    return done


def _chunks(items: Sequence[BirdExample], size: int) -> Iterator[Sequence[BirdExample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_resumable(
    examples: Sequence[BirdExample],
    predict: Callable[[Sequence[BirdExample]], tuple[list[dict[str, Any]], dict[str, Any]]],
    progress_path: Path | str,
    *,
    chunk_size: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate in chunks, checkpointing to ``progress_path`` after each one.

    ``predict`` is called with the examples still outstanding in a chunk and must
    return the ``(records, meta)`` pair :func:`sqlpup.eval.predict.
    generate_predictions` returns. The meta reported back is the last chunk's,
    with the run-level counts corrected to cover every example.
    """
    progress = Path(progress_path)
    progress.parent.mkdir(parents=True, exist_ok=True)
    done = _read_progress(progress)
    if done:
        logger.info("resuming: %d of %d examples already predicted", len(done), len(examples))

    meta: dict[str, Any] = {}
    completed = 0
    for number, chunk in enumerate(_chunks(examples, chunk_size), start=1):
        outstanding = [example for example in chunk if example.index not in done]
        if not outstanding:
            logger.info("chunk %d: already complete, skipping", number)
            continue
        logger.info(
            "chunk %d: predicting %d examples (%d/%d done)",
            number,
            len(outstanding),
            len(done),
            len(examples),
        )
        records, meta = predict(outstanding)
        with progress.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                done[int(record["index"])] = record
        completed += len(records)

    ordered = [done[example.index] for example in examples if example.index in done]
    meta = {**meta, "examples": len(ordered), "generated_this_run": completed}
    logger.info("finished: %d predictions for %d examples", len(ordered), len(examples))
    return ordered, meta
