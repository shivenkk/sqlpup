"""The verdict panel: one function, every pre-registered v3 diagnostic.

Each metric here answers a gate written down before the numbers existed
(ops audit doc, reviews #9-#14): does the SQL body honour the linking block
the model itself wrote (exposure bias), how does EX split by difficulty and
by JOIN count (relational-reasoning ceiling), what fraction of predictions
execute at all (the raw GRPO ignition gate), and how well are identifiers
bound (linking F1).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlpup.eval.dataset import BirdExample
from sqlpup.eval.diagnose import diagnose_predictions


def _eval_dir(tmp_path: Path) -> Path:
    eval_dir = tmp_path / "eval"
    db_dir = eval_dir / "dev_databases" / "toy"
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / "toy.sqlite")
    con.executescript(
        "CREATE TABLE t (id INTEGER, name TEXT);"
        "CREATE TABLE u (id INTEGER, t_id INTEGER, city TEXT);"
        "INSERT INTO t VALUES (1,'a'),(2,'b');"
        "INSERT INTO u VALUES (1,1,'x'),(2,2,'y');"
    )
    con.commit()
    con.close()
    return eval_dir


def _score(verdicts: list[tuple[str, bool]]) -> dict[str, object]:
    """Minimal `eval score` output: one (category, match) per example."""
    return {
        "examples": [
            {"index": i, "category": category, "match": match}
            for i, (category, match) in enumerate(verdicts)
        ]
    }


def _examples() -> list[BirdExample]:
    return [
        BirdExample(
            index=0,
            question_id="0",
            db_id="toy",
            question="names?",
            evidence="",
            gold_sql="SELECT name FROM t",
            difficulty="simple",
        ),
        BirdExample(
            index=1,
            question_id="1",
            db_id="toy",
            question="cities?",
            evidence="",
            gold_sql="SELECT u.city FROM u JOIN t ON u.t_id = t.id",
            difficulty="moderate",
        ),
        BirdExample(
            index=2,
            question_id="2",
            db_id="toy",
            question="broken?",
            evidence="",
            gold_sql="SELECT id FROM t",
            difficulty="challenging",
        ),
    ]


def test_panel_reports_difficulty_join_and_validity_splits(tmp_path: Path) -> None:
    eval_dir = _eval_dir(tmp_path)
    predictions = [
        {"index": 0, "predicted_sql": "SELECT name FROM t"},  # correct, 0 joins
        {
            "index": 1,
            "predicted_sql": "SELECT u.city FROM u JOIN t ON u.t_id = t.id",
        },  # correct, 1 join
        {"index": 2, "predicted_sql": "SELECT nope FROM t"},  # invalid
    ]
    score = _score([("ok", True), ("ok", True), ("execution_error", False)])
    panel = diagnose_predictions(predictions, _examples(), eval_dir, score=score)

    assert panel["ex"] == 2 / 3
    assert panel["by_difficulty"]["simple"]["ex"] == 1.0
    assert panel["by_difficulty"]["challenging"]["ex"] == 0.0
    assert panel["by_joins"]["0-1"]["total"] == 3  # gold join counts
    assert panel["valid_sql_rate"] == 2 / 3  # the third does not execute
    assert panel["linking_f1"] > 0.6  # identifiers mostly right


def test_panel_measures_block_vs_sql_divergence(tmp_path: Path) -> None:
    eval_dir = _eval_dir(tmp_path)
    predictions = [
        {  # block names t, SQL uses t -> consistent
            "index": 0,
            "predicted_sql": "SELECT name FROM t",
            "raw_completion": "-- tables: t\n-- columns: t.name\nSELECT name FROM t",
        },
        {  # block names u, SQL queries t -> divergent (exposure-bias signal)
            "index": 1,
            "predicted_sql": "SELECT city FROM t",
            "raw_completion": "-- tables: u\n-- columns: u.city\nSELECT city FROM t",
        },
        {  # block invents a table that is not in the schema
            "index": 2,
            "predicted_sql": "SELECT id FROM ghosts",
            "raw_completion": "-- tables: ghosts\n-- columns: ghosts.id\nSELECT id FROM ghosts",
        },
    ]
    score = _score([("ok", True), ("ok", False), ("execution_error", False)])
    panel = diagnose_predictions(predictions, _examples(), eval_dir, score=score)
    block = panel["block"]
    assert block["with_block"] == 3
    assert block["divergent"] == 1  # example 1 only
    assert block["hallucinated_identifiers"] == 1  # example 2's ghosts
    assert 0.0 < block["divergence_rate"] < 1.0


def test_panel_tolerates_predictions_without_blocks(tmp_path: Path) -> None:
    eval_dir = _eval_dir(tmp_path)
    predictions = [{"index": i, "predicted_sql": "SELECT 1"} for i in range(3)]
    score = _score([("ok", False)] * 3)
    panel = diagnose_predictions(predictions, _examples(), eval_dir, score=score)
    assert panel["block"]["with_block"] == 0
    assert panel["block"]["divergence_rate"] is None
