"""Prediction generation: prompts from real schemas, provenance-carrying records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from sqlpup.cli import main
from sqlpup.eval.dataset import load_examples
from sqlpup.eval.execution import ExecutionScorer
from sqlpup.eval.generate import FakeGenerator
from sqlpup.eval.predict import generate_predictions


def _write_eval_dir(tmp_path: Path) -> Path:
    """Two-example mini-dev layout over one directly-laid-out database."""
    eval_dir = tmp_path / "eval"
    db_dir = eval_dir / "dev_databases" / "toy"
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / "toy.sqlite")
    con.executescript(
        "CREATE TABLE t (id INTEGER, name TEXT);INSERT INTO t VALUES (1, 'alpha'), (2, 'beta');"
    )
    con.commit()
    con.close()
    examples = [
        {
            "question_id": i,
            "db_id": "toy",
            "question": q,
            "evidence": e,
            "SQL": "SELECT id FROM t",
            "difficulty": "simple",
        }
        for i, (q, e) in enumerate([("how many rows?", "count means COUNT(*)"), ("names?", "")])
    ]
    (eval_dir / "mini_dev_sqlite.json").write_text(json.dumps(examples), encoding="utf-8")
    return eval_dir


def test_generate_predictions_renders_real_schema_and_aligns_records(tmp_path: Path) -> None:
    eval_dir = _write_eval_dir(tmp_path)
    examples = load_examples(eval_dir, "mini-dev")
    fake = FakeGenerator(script=[["SELECT COUNT(*) FROM t;", "SELECT name FROM t;"]])

    records, meta = generate_predictions(examples, eval_dir, fake)

    prompt = fake.calls[0][0]
    assert "CREATE TABLE t (id INTEGER, name TEXT);" in prompt  # real DDL, verbatim
    assert "-- External knowledge: count means COUNT(*)" in prompt
    assert "-- Question: how many rows?" in prompt
    assert fake.calls[0][1].count("External knowledge") == 0  # blank evidence dropped

    assert records == [
        {
            "index": 0,
            "predicted_sql": "SELECT COUNT(*) FROM t",
            "raw_completion": "SELECT COUNT(*) FROM t;",
        },
        {
            "index": 1,
            "predicted_sql": "SELECT name FROM t",
            "raw_completion": "SELECT name FROM t;",
        },
    ]
    assert meta["prompt_spec"].startswith("bird-ddl-v1@")
    assert meta["generator"] == "FakeGenerator"
    assert meta["refine_retries"] is None
    assert meta["examples"] == 2


def test_generate_predictions_refined_path_records_both_sqls(tmp_path: Path) -> None:
    eval_dir = _write_eval_dir(tmp_path)
    examples = load_examples(eval_dir, "mini-dev")
    fake = FakeGenerator(
        script=[["SELECT missing FROM t", "SELECT name FROM t"], ["SELECT id FROM t"]]
    )

    from sqlpup.eval import ExecutionScorer

    with ExecutionScorer() as scorer:
        records, meta = generate_predictions(
            examples, eval_dir, fake, refine_retries=2, scorer=scorer
        )

    assert records[0]["predicted_sql"] == "SELECT id FROM t"  # repaired
    assert records[0]["single_shot_sql"] == "SELECT missing FROM t"
    assert records[1]["predicted_sql"] == "SELECT name FROM t"
    assert records[1]["single_shot_sql"] == "SELECT name FROM t"
    assert meta["refine_retries"] == 2


def test_generate_predictions_refine_requires_a_scorer(tmp_path: Path) -> None:
    eval_dir = _write_eval_dir(tmp_path)
    examples = load_examples(eval_dir, "mini-dev")
    with pytest.raises(ValueError):
        generate_predictions(examples, eval_dir, FakeGenerator(script=[]), refine_retries=1)


def test_eval_generate_cli_writes_predictions_and_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = _write_eval_dir(tmp_path)
    out = tmp_path / "preds.json"

    import sqlpup.eval.hf_generator as hf_generator

    def _fake_factory(model_dir: object, **kwargs: object) -> FakeGenerator:
        return FakeGenerator(script=[["SELECT COUNT(*) FROM t;", "SELECT name FROM t;"]])

    monkeypatch.setattr(hf_generator, "HFGreedyGenerator", _fake_factory)

    rc = main(
        [
            "eval",
            "generate",
            "--model-dir",
            "unused-by-fake",
            "--subset",
            "mini-dev",
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(out),
        ]
    )
    assert rc == 0

    records = json.loads(out.read_text(encoding="utf-8"))
    assert [r["index"] for r in records] == [0, 1]
    assert records[0]["predicted_sql"] == "SELECT COUNT(*) FROM t"
    meta = json.loads((tmp_path / "preds.json.meta.json").read_text(encoding="utf-8"))
    assert meta["prompt_spec"].startswith("bird-ddl-v1@")
    assert meta["subset"] == "mini-dev"


def test_eval_generate_cli_limit_slices_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_dir = _write_eval_dir(tmp_path)
    out = tmp_path / "preds.json"

    import sqlpup.eval.hf_generator as hf_generator

    def _fake_factory(model_dir: object, **kwargs: object) -> FakeGenerator:
        return FakeGenerator(script=[["SELECT COUNT(*) FROM t;"]])

    monkeypatch.setattr(hf_generator, "HFGreedyGenerator", _fake_factory)

    rc = main(
        [
            "eval",
            "generate",
            "--model-dir",
            "unused",
            "--subset",
            "mini-dev",
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(out),
            "--limit",
            "1",
        ]
    )
    assert rc == 0
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 1


def test_generate_predictions_prepends_completion_prefix(tmp_path: Path) -> None:
    from sqlpup.eval.prompts import BIRD_DDL_V1_SELECTCUE

    eval_dir = _write_eval_dir(tmp_path)
    examples = load_examples(eval_dir, "mini-dev")
    fake = FakeGenerator(script=[[" COUNT(*) FROM t;", " name FROM t;"]])
    records, meta = generate_predictions(examples, eval_dir, fake, spec=BIRD_DDL_V1_SELECTCUE)
    assert records[0]["predicted_sql"] == "SELECT COUNT(*) FROM t"
    assert records[1]["predicted_sql"] == "SELECT name FROM t"
    assert meta["prompt_spec"].startswith("bird-ddl-v1-selectcue@")
    assert fake.calls[0][0].endswith("-- SQL:\nSELECT")


def test_block_constrain_routes_through_constrained_generation(tmp_path: Path) -> None:
    eval_dir = _write_eval_dir(tmp_path)

    class _ConstrainedFake:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], list[Path]]] = []

        def generate(self, prompts):  # type: ignore[no-untyped-def]
            raise AssertionError("must not fall back to unconstrained generate")

        def generate_block_constrained(self, prompts, db_paths):  # type: ignore[no-untyped-def]
            self.calls.append((list(prompts), list(db_paths)))
            return ["SELECT 1"] * len(prompts)

    gen = _ConstrainedFake()
    examples = load_examples(eval_dir, "mini-dev")
    records, meta = generate_predictions(examples, eval_dir, gen, block_constrain=True)
    assert records[0]["predicted_sql"] == "SELECT 1"
    assert records[0]["raw_completion"] == "SELECT 1"
    assert meta["block_constrained"] is True
    assert len(gen.calls) == 1
    assert gen.calls[0][1][0].name == "toy.sqlite"


def test_block_constrain_refuses_generators_without_support(tmp_path: Path) -> None:
    eval_dir = _write_eval_dir(tmp_path)
    examples = load_examples(eval_dir, "mini-dev")
    with pytest.raises(NotImplementedError):
        generate_predictions(examples, eval_dir, FakeGenerator(script=[]), block_constrain=True)


def test_two_pass_narrows_the_schema_using_the_models_own_block(tmp_path: Path) -> None:
    """Pass 2 must see only the tables pass 1 named, and answer from there."""
    eval_dir = _write_eval_dir(tmp_path)
    # a second table exists so narrowing is observable
    con = sqlite3.connect(eval_dir / "dev_databases" / "toy" / "toy.sqlite")
    con.executescript("CREATE TABLE unrelated (id INTEGER, junk TEXT);")
    con.commit()
    con.close()

    fake = FakeGenerator(
        script=[
            # pass 1: blocks naming only `t`
            [
                "-- tables: t\n-- columns: t.name\nSELECT 1;",
                "-- tables: t\n-- columns: t.id\nSELECT 2;",
            ],
            # pass 2: the real answers
            ["SELECT COUNT(*) FROM t;", "SELECT name FROM t;"],
        ]
    )
    examples = load_examples(eval_dir, "mini-dev")
    records, meta = generate_predictions(examples, eval_dir, fake, two_pass=True)

    assert meta["two_pass"] is True
    assert [r["predicted_sql"] for r in records] == ["SELECT COUNT(*) FROM t", "SELECT name FROM t"]
    first_prompt, second_prompt = fake.calls[0][0], fake.calls[1][0]
    assert "unrelated" in first_prompt  # pass 1 sees the whole schema
    assert "unrelated" not in second_prompt  # pass 2 is narrowed to `t`
    assert records[0]["first_pass_sql"] == "SELECT 1"  # both framings stay reportable


def test_two_pass_falls_back_to_full_schema_when_pass_one_names_nothing(tmp_path: Path) -> None:
    eval_dir = _write_eval_dir(tmp_path)
    fake = FakeGenerator(
        script=[["SELECT 1;", "SELECT 2;"], ["SELECT COUNT(*) FROM t;", "SELECT name FROM t;"]]
    )
    examples = load_examples(eval_dir, "mini-dev")
    records, _ = generate_predictions(examples, eval_dir, fake, two_pass=True)
    assert "CREATE TABLE t" in fake.calls[1][0]  # still a real schema, not empty
    assert records[0]["predicted_sql"] == "SELECT COUNT(*) FROM t"


def test_self_consistency_votes_over_k_samples_and_keeps_them_all(tmp_path: Path) -> None:
    """k samples per prompt, majority answer submitted, every candidate kept so
    the oracle union (is the right answer even reachable?) is computable later."""
    eval_dir = _write_eval_dir(tmp_path)

    class _SamplingFake:
        """Returns k scripted samples per prompt."""

        def __init__(self, per_prompt: list[str]) -> None:
            self.per_prompt = per_prompt
            self.k_asked: int | None = None

        def generate(self, prompts: Sequence[str]) -> list[str]:
            raise AssertionError("self-consistency must request samples, not greedy")

        def generate_samples(
            self, prompts: Sequence[str], k: int, temperature: float, seed: int
        ) -> list[list[str]]:
            self.k_asked = k
            return [list(self.per_prompt) for _ in prompts]

    # two samples agree on COUNT(*), one disagrees -> majority wins
    gen = _SamplingFake(
        ["SELECT COUNT(*) FROM t;", "SELECT name FROM t;", "SELECT COUNT(*) FROM t;"]
    )
    examples = load_examples(eval_dir, "mini-dev")
    with ExecutionScorer() as scorer:
        records, meta = generate_predictions(
            examples, eval_dir, gen, self_consistency=3, scorer=scorer
        )
    assert gen.k_asked == 3
    assert meta["self_consistency"] == 3
    assert records[0]["predicted_sql"] == "SELECT COUNT(*) FROM t"
    assert records[0]["vote_stats"]["votes"] == 2
    assert len(records[0]["candidates"]) == 3  # kept for the oracle-union analysis
