"""Held-out prediction: gold-blind loading, BIRD's file format, resume after a crash."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from sqlpup.cli import main
from sqlpup.eval.dataset import BirdExample, load_prediction_examples
from sqlpup.eval.generate import FakeGenerator
from sqlpup.eval.submit import (
    BIRD_SEPARATOR,
    bird_prediction,
    run_resumable,
    write_bird_predictions,
)


def _example(index: int, db_id: str = "toy") -> BirdExample:
    return BirdExample(
        index=index,
        question_id=str(index),
        db_id=db_id,
        question="how many rows?",
        evidence="",
        gold_sql="",
        difficulty="",
    )


def _write_test_split(tmp_path: Path, *, with_empty_sql_key: bool) -> tuple[Path, Path]:
    """A test-format split: two questions over one database, no usable gold."""
    db_root = tmp_path / "test_databases"
    db_dir = db_root / "toy"
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / "toy.sqlite")
    con.executescript(
        "CREATE TABLE t (id INTEGER, name TEXT);INSERT INTO t VALUES (1, 'alpha'), (2, 'beta');"
    )
    con.commit()
    con.close()
    rows: list[dict[str, Any]] = [
        {"question_id": i, "db_id": "toy", "question": q, "evidence": ""}
        for i, q in enumerate(["how many rows?", "names?"])
    ]
    if with_empty_sql_key:
        for row in rows:
            row["SQL"] = ""
    examples_path = tmp_path / "test.json"
    examples_path.write_text(json.dumps(rows), encoding="utf-8")
    return examples_path, db_root


@pytest.mark.parametrize("with_empty_sql_key", [True, False])
def test_a_split_loads_whether_gold_is_empty_or_absent(
    tmp_path: Path, with_empty_sql_key: bool
) -> None:
    examples_path, _ = _write_test_split(tmp_path, with_empty_sql_key=with_empty_sql_key)

    examples = load_prediction_examples(examples_path)

    assert [e.index for e in examples] == [0, 1]
    assert [e.gold_sql for e in examples] == ["", ""]  # never a KeyError, never a real answer


def test_a_prediction_entry_is_one_line_so_the_separator_cannot_be_ambiguous() -> None:
    entry = bird_prediction("SELECT COUNT(*)\n  FROM t\tWHERE id > 0", "toy")

    sql, db_id = entry.split(BIRD_SEPARATOR)
    assert sql == "SELECT COUNT(*) FROM t WHERE id > 0"
    assert db_id == "toy"
    assert "\n" not in sql and "\t" not in sql


def test_the_submission_file_is_keyed_by_position_in_example_order(tmp_path: Path) -> None:
    examples = [_example(0), _example(1, "other")]
    records = [
        {"index": 1, "predicted_sql": "SELECT name FROM t"},
        {"index": 0, "predicted_sql": "SELECT COUNT(*) FROM t"},
    ]

    empties = write_bird_predictions(records, examples, tmp_path / "predict_test.json")

    payload = json.loads((tmp_path / "predict_test.json").read_text())
    assert list(payload) == ["0", "1"]  # example order, not record order
    assert payload["0"] == f"SELECT COUNT(*) FROM t{BIRD_SEPARATOR}toy"
    assert payload["1"] == f"SELECT name FROM t{BIRD_SEPARATOR}other"
    assert empties == 0


def test_a_missing_prediction_is_refused_rather_than_silently_shifting_answers(
    tmp_path: Path,
) -> None:
    examples = [_example(0), _example(1)]

    with pytest.raises(ValueError, match="misaligned"):
        write_bird_predictions(
            [{"index": 0, "predicted_sql": "SELECT 1"}], examples, tmp_path / "predict_test.json"
        )


def test_empty_predictions_are_counted_so_the_rate_can_be_reported(tmp_path: Path) -> None:
    examples = [_example(0), _example(1)]
    records = [{"index": 0, "predicted_sql": ""}, {"index": 1, "predicted_sql": "SELECT 1"}]

    assert write_bird_predictions(records, examples, tmp_path / "predict_test.json") == 1


def test_a_resumed_run_only_generates_what_is_missing(tmp_path: Path) -> None:
    examples = [_example(i) for i in range(5)]
    progress = tmp_path / "progress.jsonl"
    asked: list[list[int]] = []

    def predict(chunk: Sequence[BirdExample]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        asked.append([e.index for e in chunk])
        return [{"index": e.index, "predicted_sql": f"SELECT {e.index}"} for e in chunk], {
            "prompt_spec": "bird-ddl-v1"
        }

    first, _ = run_resumable(examples[:3], predict, progress, chunk_size=2)
    assert asked == [[0, 1], [2]]
    assert len(first) == 3

    # Same command again over the full split: the first three are on disk already.
    asked.clear()
    second, meta = run_resumable(examples, predict, progress, chunk_size=2)

    assert asked == [[3], [4]]  # chunk [0,1] and the done part of [2,3] are skipped
    assert [r["index"] for r in second] == [0, 1, 2, 3, 4]
    assert meta["generated_this_run"] == 2


def test_a_truncated_progress_line_costs_one_example_not_the_run(tmp_path: Path) -> None:
    progress = tmp_path / "progress.jsonl"
    progress.write_text('{"index": 0, "predicted_sql": "SELECT 0"}\n{"index": 1, "pred', "utf-8")
    regenerated: list[int] = []

    def predict(chunk: Sequence[BirdExample]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        regenerated.extend(e.index for e in chunk)
        return [{"index": e.index, "predicted_sql": "SELECT 1"} for e in chunk], {}

    records, _ = run_resumable([_example(0), _example(1)], predict, progress, chunk_size=2)

    assert regenerated == [1]
    assert [r["index"] for r in records] == [0, 1]


def test_the_cli_predicts_a_gold_free_split_end_to_end(tmp_path: Path) -> None:
    examples_path, db_root = _write_test_split(tmp_path, with_empty_sql_key=True)
    out_dir = tmp_path / "submission"
    fake = FakeGenerator(script=[["SELECT COUNT(*) FROM t;", "SELECT name FROM t;"]])

    import sqlpup.eval.hf_generator as hf_generator

    original = hf_generator.HFGreedyGenerator
    hf_generator.HFGreedyGenerator = lambda *a, **k: fake  # type: ignore[assignment,misc]
    try:
        code = main(
            [
                "eval",
                "predict",
                "--model-dir",
                "unused",
                "--examples",
                str(examples_path),
                "--db-root",
                str(db_root),
                "--out-dir",
                str(out_dir),
            ]
        )
    finally:
        hf_generator.HFGreedyGenerator = original  # type: ignore[misc]

    assert code == 0
    payload = json.loads((out_dir / "predict_test.json").read_text())
    assert payload == {  # extraction drops the trailing semicolon
        "0": f"SELECT COUNT(*) FROM t{BIRD_SEPARATOR}toy",
        "1": f"SELECT name FROM t{BIRD_SEPARATOR}toy",
    }
    assert "CREATE TABLE t (id INTEGER, name TEXT);" in fake.calls[0][0]  # real DDL, from test dbs
    assert (out_dir / "run.log").read_text()  # a third party can see what happened
    meta = json.loads((out_dir / "records.json.meta.json").read_text())
    assert meta["split"] == "test"
    assert meta["empty_predictions"] == 0


def test_a_missing_database_fails_before_the_model_is_loaded(tmp_path: Path) -> None:
    examples_path, db_root = _write_test_split(tmp_path, with_empty_sql_key=True)
    rows = json.loads(examples_path.read_text())
    rows[0]["db_id"] = "absent"
    examples_path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(SystemExit, match="absent"):
        main(
            [
                "eval",
                "predict",
                "--model-dir",
                "unused",
                "--examples",
                str(examples_path),
                "--db-root",
                str(db_root),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
