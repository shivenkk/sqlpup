"""Batch scoring: EX overall + per-difficulty breakdown, and predictions loading.

Uses the real ExecutionScorer against a tiny in-process SQLite database so the
arithmetic is checked end to end on hand-computed fixtures.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sqlpup.eval import ExecutionScorer
from sqlpup.eval.dataset import BirdExample, resolve_db_path
from sqlpup.eval.scorer import EvalReport, load_predictions, score_predictions


@pytest.fixture
def eval_dir(tmp_path: Path) -> Path:
    db_path = resolve_db_path(tmp_path, "fix")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE t (id INTEGER)")
    con.executemany("INSERT INTO t (id) VALUES (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()
    return tmp_path


def _example(index: int, difficulty: str, gold_sql: str) -> BirdExample:
    return BirdExample(
        index=index,
        question_id=str(index),
        db_id="fix",
        question=f"q{index}",
        evidence="",
        gold_sql=gold_sql,
        difficulty=difficulty,
    )


def test_score_predictions_ex_and_difficulty_breakdown(eval_dir: Path) -> None:
    examples = [
        _example(0, "simple", "SELECT id FROM t"),
        _example(1, "simple", "SELECT id FROM t"),
        _example(2, "simple", "SELECT id FROM t"),
        _example(3, "moderate", "SELECT id FROM t WHERE id = 1"),
        _example(4, "moderate", "SELECT id FROM t"),
        _example(5, "challenging", "SELECT id FROM t"),
    ]
    predictions = {
        0: "SELECT id FROM t",  # exact -> match (ok)
        1: "SELECT id FROM t ORDER BY id DESC",  # reordered -> match (ok)
        2: "SELECT id FROM t WHERE id = 1",  # wrong rows -> non-match (ok)
        3: "SELECT id FROM t WHERE id = 1",  # match (ok)
        # index 4 has no prediction -> missing, non-match
        5: "SELECT * FROM nope",  # error -> non-match (execution_error)
    }

    with ExecutionScorer() as scorer:
        report = score_predictions(examples, predictions, scorer, eval_dir, subset="dev")

    assert isinstance(report, EvalReport)
    assert report.total == 6
    assert report.correct == 3
    assert report.ex == pytest.approx(0.5)

    by_diff = {d.difficulty: d for d in report.by_difficulty}
    assert (by_diff["simple"].total, by_diff["simple"].correct) == (3, 2)
    assert by_diff["simple"].ex == pytest.approx(2 / 3)
    assert (by_diff["moderate"].total, by_diff["moderate"].correct) == (2, 1)
    assert by_diff["moderate"].ex == pytest.approx(0.5)
    assert (by_diff["challenging"].total, by_diff["challenging"].correct) == (1, 0)
    assert by_diff["challenging"].ex == pytest.approx(0.0)

    assert report.missing == 1
    assert report.category_counts["ok"] == 4
    assert report.category_counts["execution_error"] == 1
    assert report.category_counts["missing"] == 1


def test_score_predictions_all_correct_is_exactly_one(eval_dir: Path) -> None:
    # The acceptance-gate shape in miniature: gold as predictions -> EX 1.0.
    examples = [
        _example(0, "simple", "SELECT id FROM t"),
        _example(1, "moderate", "SELECT id FROM t WHERE id = 2"),
        _example(2, "challenging", "SELECT id FROM t WHERE id > 1"),
    ]
    predictions = {e.index: e.gold_sql for e in examples}
    with ExecutionScorer() as scorer:
        report = score_predictions(examples, predictions, scorer, eval_dir, subset="dev")
    assert report.ex == 1.0
    assert all(d.ex == 1.0 for d in report.by_difficulty)


def test_report_summary_and_detail_dicts(eval_dir: Path) -> None:
    examples = [_example(0, "simple", "SELECT id FROM t")]
    with ExecutionScorer() as scorer:
        report = score_predictions(
            examples, {0: "SELECT id FROM t"}, scorer, eval_dir, subset="dev"
        )
    summary = report.summary_dict()
    assert summary["subset"] == "dev"
    assert summary["total"] == 1
    assert summary["ex"] == 1.0
    assert summary["by_difficulty"]["simple"]["ex"] == 1.0
    # summary is JSON-serialisable for _emit
    assert json.loads(json.dumps(summary))["ex"] == 1.0

    detail = report.detail_dict()
    assert detail["summary"] == summary
    assert detail["examples"][0]["index"] == 0
    assert detail["examples"][0]["match"] is True
    assert detail["examples"][0]["category"] == "ok"


# --- predictions loading -----------------------------------------------------


def test_load_predictions_json_array_keyed_by_index(tmp_path: Path) -> None:
    path = tmp_path / "preds.json"
    path.write_text(
        json.dumps(
            [
                {"db_id": "fix", "predicted_sql": "SELECT 1"},
                {"index": 5, "db_id": "fix", "predicted_sql": "SELECT 2"},
            ]
        ),
        encoding="utf-8",
    )
    preds = load_predictions(path)
    assert preds == {0: "SELECT 1", 5: "SELECT 2"}


def test_load_predictions_jsonl_positional(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    path.write_text(
        '{"predicted_sql": "SELECT 1"}\n{"predicted_sql": "SELECT 2"}\n',
        encoding="utf-8",
    )
    preds = load_predictions(path)
    assert preds == {0: "SELECT 1", 1: "SELECT 2"}


def test_load_predictions_rejects_non_string_sql(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"predicted_sql": 123}]), encoding="utf-8")
    with pytest.raises(ValueError, match="predicted_sql"):
        load_predictions(path)
