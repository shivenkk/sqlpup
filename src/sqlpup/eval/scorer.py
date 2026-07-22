"""Batch scoring over a predictions file: BIRD execution accuracy (EX).

Sequential by design (addendum: one worker, no fan-out) -- the per-query timeout
bounds total runtime. EX is the fraction of predictions whose result set equals
gold's, reported overall and per BIRD difficulty bucket (simple / moderate /
challenging), exactly as the official report breaks it down. Deterministic:
examples are scored in load order and every emitted mapping is sorted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlpup.eval.dataset import DIFFICULTIES, BirdExample, resolve_db_path
from sqlpup.eval.execution import ExecutionScorer

# Scorer-level category for an example that has no prediction at all (scores 0,
# like the official evaluator's unparseable/absent prediction).
MISSING = "missing"


def _ex(correct: int, total: int) -> float:
    return correct / total if total else 0.0


@dataclass(frozen=True, slots=True)
class ExampleResult:
    """Per-example verdict retained for the detailed (gitignored) artifact."""

    index: int
    question_id: str
    db_id: str
    difficulty: str
    match: bool
    category: str


@dataclass(frozen=True, slots=True)
class DifficultyStat:
    difficulty: str
    total: int
    correct: int

    @property
    def ex(self) -> float:
        return _ex(self.correct, self.total)


@dataclass(frozen=True, slots=True)
class EvalReport:
    subset: str
    total: int
    correct: int
    by_difficulty: tuple[DifficultyStat, ...]
    category_counts: Mapping[str, int]
    missing: int
    results: tuple[ExampleResult, ...]

    @property
    def ex(self) -> float:
        return _ex(self.correct, self.total)

    def summary_dict(self) -> dict[str, Any]:
        """The stdout summary (overall EX + per-difficulty + category counts)."""
        return {
            "subset": self.subset,
            "total": self.total,
            "correct": self.correct,
            "ex": self.ex,
            "missing": self.missing,
            "by_difficulty": {
                stat.difficulty: {
                    "total": stat.total,
                    "correct": stat.correct,
                    "ex": stat.ex,
                }
                for stat in self.by_difficulty
            },
            "categories": dict(sorted(self.category_counts.items())),
        }

    def detail_dict(self) -> dict[str, Any]:
        """The full per-example artifact written to ``--out`` (gitignored)."""
        return {
            "summary": self.summary_dict(),
            "examples": [
                {
                    "index": r.index,
                    "question_id": r.question_id,
                    "db_id": r.db_id,
                    "difficulty": r.difficulty,
                    "match": r.match,
                    "category": r.category,
                }
                for r in self.results
            ],
        }


def score_predictions(
    examples: Sequence[BirdExample],
    predictions: Mapping[int, str],
    scorer: ExecutionScorer,
    eval_dir: Path,
    subset: str = "dev",
) -> EvalReport:
    """Score every example against its prediction, sequentially, and tally EX."""
    results: list[ExampleResult] = []
    category_counts: dict[str, int] = {}
    diff_total: dict[str, int] = dict.fromkeys(DIFFICULTIES, 0)
    diff_correct: dict[str, int] = dict.fromkeys(DIFFICULTIES, 0)
    correct = 0
    missing = 0

    for example in examples:
        predicted_sql = predictions.get(example.index)
        if predicted_sql is None:
            match, category = False, MISSING
            missing += 1
        else:
            db_path = resolve_db_path(eval_dir, example.db_id)
            verdict = scorer.score(predicted_sql, example.gold_sql, db_path)
            match, category = verdict.match, verdict.category.value

        category_counts[category] = category_counts.get(category, 0) + 1
        if match:
            correct += 1
        if example.difficulty in diff_total:
            diff_total[example.difficulty] += 1
            if match:
                diff_correct[example.difficulty] += 1
        results.append(
            ExampleResult(
                index=example.index,
                question_id=example.question_id,
                db_id=example.db_id,
                difficulty=example.difficulty,
                match=match,
                category=category,
            )
        )

    by_difficulty = tuple(
        DifficultyStat(difficulty=d, total=diff_total[d], correct=diff_correct[d])
        for d in DIFFICULTIES
    )
    return EvalReport(
        subset=subset,
        total=len(examples),
        correct=correct,
        by_difficulty=by_difficulty,
        category_counts=category_counts,
        missing=missing,
        results=tuple(results),
    )


def _parse_entries(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":  # a JSON array of prediction objects
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array of predictions")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]  # JSONL


def load_predictions(path: Path) -> dict[int, str]:
    """Load predictions keyed by 0-based example ``index``.

    Accepts a JSON array or JSONL of objects, each with ``predicted_sql`` (and an
    optional ``index`` overriding the positional one; an optional ``db_id`` is
    accepted but ignored, since execution always uses the example's database).
    """
    out: dict[int, str] = {}
    for position, entry in enumerate(_parse_entries(path)):
        if not isinstance(entry, Mapping):
            raise ValueError(f"prediction {position}: expected a JSON object")
        index = int(entry["index"]) if "index" in entry else position
        predicted_sql = entry.get("predicted_sql")
        if not isinstance(predicted_sql, str):
            raise ValueError(f"prediction {position}: 'predicted_sql' must be a string")
        out[index] = predicted_sql
    return out
