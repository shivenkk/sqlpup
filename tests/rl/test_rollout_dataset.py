"""Building the GRPO rollout dataset.

Three properties matter, and each has a silent-failure mode:

* **Prompts must be rendered by the same PromptSpec evaluation uses.** A
  train/eval prompt mismatch would teach the model a format it is never scored
  under, and nothing in the loss would reveal it.
* **Every prompt must leave room to answer.** TRL 1.9 has no
  ``max_prompt_length``; a prompt occupying the whole context yields truncated
  rollouts, which score 0 regardless of what the model knows, so the gradient
  is noise. We reuse the measured schema-compaction ladder to guarantee room.
* **Rows must carry gold SQL and a database path**, because the reward is
  execution-based and TRL forwards dataset columns to it verbatim.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlpup.eval.dataset import BirdExample
from sqlpup.rl.rollout_data import build_rollout_rows


def _db_root(tmp_path: Path, columns: int = 2, db_id: str = "toy") -> Path:
    root = tmp_path / "dbs"
    (root / db_id).mkdir(parents=True)
    con = sqlite3.connect(root / db_id / f"{db_id}.sqlite")
    body = ", ".join(f"column_number_{i} TEXT DEFAULT 'x'" for i in range(columns))
    con.execute(f"CREATE TABLE t (id INTEGER PRIMARY KEY, {body})")
    con.commit()
    con.close()
    return root


def _example(index: int = 0, db_id: str = "toy") -> BirdExample:
    return BirdExample(
        index=index,
        question_id=f"q{index}",
        db_id=db_id,
        question="how many rows?",
        evidence="",
        gold_sql="SELECT COUNT(*) FROM t",
        difficulty="simple",
    )


def test_rows_carry_exactly_what_trl_and_the_reward_need(tmp_path: Path) -> None:
    root = _db_root(tmp_path)
    rows, _ = build_rollout_rows(
        [_example()], root, count_tokens=len, context_limit=10_000, max_completion_length=256
    )
    assert len(rows) == 1
    assert set(rows[0]) == {"prompt", "gold_sql", "db_path"}
    assert rows[0]["gold_sql"] == "SELECT COUNT(*) FROM t"
    assert Path(rows[0]["db_path"]).exists()
    assert "CREATE TABLE" in rows[0]["prompt"] and "how many rows?" in rows[0]["prompt"]


def test_prompt_matches_the_evaluation_prompt_spec_byte_for_byte(tmp_path: Path) -> None:
    """If these ever diverge, the model is trained on one format and scored on
    another, and no metric in the training loop would show it."""
    from sqlpup.eval.prompts import BIRD_DDL_V1, schema_ddl

    root = _db_root(tmp_path)
    rows, _ = build_rollout_rows(
        [_example()], root, count_tokens=len, context_limit=10_000, max_completion_length=256
    )
    expected = BIRD_DDL_V1.render(
        schema_ddl(root / "toy" / "toy.sqlite"), question="how many rows?", evidence=""
    )
    assert rows[0]["prompt"] == expected


def test_a_prompt_with_no_room_to_answer_is_compacted(tmp_path: Path) -> None:
    """TRL 1.9 dropped max_prompt_length, so nothing else protects this."""
    root = _db_root(tmp_path, columns=400)
    rows, stats = build_rollout_rows(
        [_example()], root, count_tokens=len, context_limit=4_000, max_completion_length=1_000
    )
    assert stats["compacted"] == 1
    assert len(rows[0]["prompt"]) <= 4_000 - 1_000


def test_a_prompt_that_cannot_fit_even_compacted_is_dropped_and_counted(tmp_path: Path) -> None:
    """Dropped rather than truncated: a half-schema prompt trains the model to
    answer from a schema it will never see at evaluation."""
    root = _db_root(tmp_path, columns=400)
    rows, stats = build_rollout_rows(
        [_example()], root, count_tokens=len, context_limit=200, max_completion_length=150
    )
    assert rows == []
    assert stats["dropped"] == 1


def test_examples_whose_database_is_missing_are_dropped_not_crashed(tmp_path: Path) -> None:
    root = _db_root(tmp_path)
    rows, stats = build_rollout_rows(
        [_example(), _example(index=1, db_id="absent")],
        root,
        count_tokens=len,
        context_limit=10_000,
        max_completion_length=256,
    )
    assert len(rows) == 1
    assert stats["missing_db"] == 1


def test_untouched_prompts_are_not_compacted(tmp_path: Path) -> None:
    root = _db_root(tmp_path)
    _, stats = build_rollout_rows(
        [_example()], root, count_tokens=len, context_limit=10_000, max_completion_length=256
    )
    assert stats["compacted"] == 0
    assert stats["kept"] == 1


@pytest.mark.parametrize("bad", [0, -1])
def test_a_nonsense_completion_budget_is_rejected(tmp_path: Path, bad: int) -> None:
    root = _db_root(tmp_path)
    with pytest.raises(ValueError):
        build_rollout_rows(
            [_example()], root, count_tokens=len, context_limit=2048, max_completion_length=bad
        )
