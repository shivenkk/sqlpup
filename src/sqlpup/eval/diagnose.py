"""The verdict panel -- every pre-registered diagnostic in one pass.

Written before v3's numbers existed so the reading of them stays mechanical
(the gates live in the ops audit doc; this module measures them):

* **block consistency** -- v3 models write a ``-- tables:/-- columns:`` block
  before the SQL. Two failure modes matter: the block naming identifiers the
  schema does not have, and the SQL body ignoring the block it just wrote
  (exposure bias). Both are counted here, from the raw completion.
* **difficulty and JOIN splits** -- the relational-reasoning ceiling shows up
  as EX collapsing with join count, not as a lower headline number.
* **valid-SQL rate** -- the raw (un-repaired) execution rate that gates GRPO.
* **linking F1** -- identifier binding measured against the gold SQL.

The panel **never executes SQL itself**. It reads the categories and match
flags that ``eval score`` already produced in its killable official sandbox,
so every rate here agrees with the number of record by construction. (A first
cut did re-execute under a cheaper budget and disagreed with the scorer on
validity -- 29% vs 36.8% on v2e -- which is exactly the class of quiet
divergence a second implementation invites.)
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sqlpup.eval.dataset import BirdExample, resolve_db_path
from sqlpup.sft.linking import schema_identifiers, sql_identifier_tokens

_JOIN = re.compile(r"\bJOIN\b", re.IGNORECASE)
_BLOCK_LINE = re.compile(r"^--\s*(tables|columns):\s*(.*)$", re.IGNORECASE)
# Scorer categories that mean "this SQL ran" (timeout = ran, took too long).
_EXECUTED = frozenset({"ok", "timeout"})


def _block_identifiers(raw: str) -> tuple[set[str], set[str]] | None:
    """(tables, ``table.column``) named by the leading linking block."""
    tables: set[str] = set()
    columns: set[str] = set()
    found = False
    for line in raw.splitlines():
        match = _BLOCK_LINE.match(line.strip())
        if match is None:
            break
        found = True
        names = {n.strip().lower() for n in match.group(2).split(",") if n.strip()}
        (tables if match.group(1).lower() == "tables" else columns).update(names)
    return (tables, columns) if found else None


def _f1(gold: set[str], pred: set[str]) -> float:
    if not gold:
        return 0.0
    hits = len(gold & pred)
    precision = hits / len(pred) if pred else 0.0
    recall = hits / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def diagnose_predictions(
    predictions: Sequence[Mapping[str, Any]],
    examples: Sequence[BirdExample],
    eval_dir: Path,
    *,
    score: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the full panel from a predictions file plus its score file."""
    by_index = {int(p["index"]): p for p in predictions}
    scored = {int(e["index"]): e for e in score["examples"]}
    schemas: dict[str, dict[str, list[str]]] = {}

    correct = valid = 0
    difficulty: dict[str, dict[str, int]] = {}
    joins: dict[str, dict[str, int]] = {
        "0-1": {"correct": 0, "total": 0},
        "2+": {"correct": 0, "total": 0},
    }
    f1s: list[float] = []
    with_block = divergent = hallucinated = 0

    for example in examples:
        prediction = by_index.get(example.index, {})
        sql = str(prediction.get("predicted_sql", ""))
        db_path = resolve_db_path(eval_dir, example.db_id)
        if example.db_id not in schemas:
            schemas[example.db_id] = schema_identifiers(db_path)
        schema = schemas[example.db_id]

        verdict = scored.get(example.index, {})
        is_correct = bool(verdict.get("match"))
        correct += int(is_correct)
        valid += int(str(verdict.get("category", "")) in _EXECUTED)

        tier = difficulty.setdefault(example.difficulty, {"correct": 0, "total": 0})
        tier["total"] += 1
        tier["correct"] += int(is_correct)

        bucket = "0-1" if len(_JOIN.findall(example.gold_sql)) <= 1 else "2+"
        joins[bucket]["total"] += 1
        joins[bucket]["correct"] += int(is_correct)

        schema_names = {t.lower() for t in schema} | {
            f"{t}.{c}".lower() for t, cols in schema.items() for c in cols
        }
        gold_ids = sql_identifier_tokens(example.gold_sql) & schema_names
        pred_ids = sql_identifier_tokens(sql) & schema_names
        if gold_ids:
            f1s.append(_f1(gold_ids, pred_ids))

        block = _block_identifiers(str(prediction.get("raw_completion", "")))
        if block is not None:
            with_block += 1
            named_tables, named_columns = block
            if (named_tables | named_columns) - schema_names:
                hallucinated += 1
            sql_tables = {t.lower() for t in schema if t.lower() in sql_identifier_tokens(sql)}
            if named_tables and sql_tables and named_tables != sql_tables:
                divergent += 1

    total = len(examples)
    return {
        "total": total,
        "ex": correct / total if total else 0.0,
        "valid_sql_rate": valid / total if total else 0.0,
        "linking_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "by_difficulty": {
            name: {**counts, "ex": counts["correct"] / counts["total"]}
            for name, counts in sorted(difficulty.items())
        },
        "by_joins": {
            name: {**counts, "ex": counts["correct"] / counts["total"] if counts["total"] else 0.0}
            for name, counts in joins.items()
        },
        "block": {
            "with_block": with_block,
            "divergent": divergent,
            "hallucinated_identifiers": hallucinated,
            "divergence_rate": divergent / with_block if with_block else None,
        },
    }
