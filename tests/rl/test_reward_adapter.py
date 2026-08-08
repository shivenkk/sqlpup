"""The TRL-shaped reward callable.

TRL invokes a custom reward as
``reward_func(prompts=..., completions=..., completion_ids=..., **columns)``
where ``columns`` carries every non-reserved dataset column as a list aligned
to the ``B*G`` rollouts, plus injected keys TRL adds itself (``trainer_state``,
``log_extra``, ``log_metric``). Getting any part of that contract wrong fails
*silently*: rewards come back all-zero, training proceeds, loss moves, and
nothing looks broken until evaluation hours later.

The specific trap for this model: v3 completions open with a
``-- tables: / -- columns:`` linking block, so a reward that scores the raw
completion scores a comment and always returns 0.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlpup.eval import ExecutionScorer
from sqlpup.rl.reward import EXECUTES_BONUS, make_execution_reward


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "f.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE t (id INTEGER, name TEXT);"
        "INSERT INTO t (id, name) VALUES (1, 'alpha'), (2, 'beta');"
    )
    con.commit()
    con.close()
    return path


GOLD = "SELECT id FROM t ORDER BY id"
BLOCK = "-- tables: t\n-- columns: t.id\n"


def test_a_completion_carrying_its_linking_block_is_still_scored(db: Path) -> None:
    """The silent killer: v3 always emits the block first. Scoring the raw
    completion would score a comment, so every reward would be 0 and the run
    would burn GPU hours learning nothing."""
    with ExecutionScorer() as scorer:
        reward = make_execution_reward(scorer)
        out = reward(
            prompts=["q"],
            completions=[BLOCK + "SELECT id FROM t"],
            completion_ids=[[0]],
            gold_sql=[GOLD],
            db_path=[str(db)],
        )
    assert out == [1.0]


def test_returns_one_float_per_rollout_in_order(db: Path) -> None:
    completions = [
        BLOCK + "SELECT id FROM t",  # correct
        BLOCK + "SELECT name FROM t",  # grounded, wrong
        BLOCK + "SELECT 1",  # degenerate
        "",  # empty
    ]
    with ExecutionScorer() as scorer:
        reward = make_execution_reward(scorer)
        out = reward(
            prompts=["q"] * 4,
            completions=completions,
            completion_ids=[[0]] * 4,
            gold_sql=[GOLD] * 4,
            db_path=[str(db)] * 4,
        )
    assert out == [1.0, EXECUTES_BONUS, 0.0, 0.0]
    assert all(isinstance(x, float) for x in out)


def test_tolerates_the_extra_kwargs_trl_injects(db: Path) -> None:
    """TRL adds trainer_state/log_extra/log_metric to the call. A reward with a
    strict signature raises TypeError mid-step, killing the run."""
    with ExecutionScorer() as scorer:
        reward = make_execution_reward(scorer)
        out = reward(
            prompts=["q"],
            completions=[BLOCK + "SELECT id FROM t"],
            completion_ids=[[0]],
            gold_sql=[GOLD],
            db_path=[str(db)],
            trainer_state=object(),
            log_extra=lambda **_: None,
            log_metric=lambda **_: None,
            difficulty=["simple"],
        )
    assert out == [1.0]


def test_a_single_bad_row_cannot_kill_a_multi_hour_run(tmp_path: Path, db: Path) -> None:
    """One unreadable database must not raise out of the reward and abort
    training. It scores 0 and is counted, so a monitor can alarm if the rate is
    material rather than the failure passing unnoticed."""
    with ExecutionScorer() as scorer:
        reward = make_execution_reward(scorer)
        out = reward(
            prompts=["q", "q"],
            completions=[BLOCK + "SELECT id FROM t", BLOCK + "SELECT id FROM t"],
            completion_ids=[[0], [0]],
            gold_sql=[GOLD, GOLD],
            db_path=[str(db), str(tmp_path / "does-not-exist.sqlite")],
        )
    assert out[0] == 1.0
    assert out[1] == 0.0
    assert reward.errors == 1


def test_it_reports_a_name_trl_can_log_under(db: Path) -> None:
    with ExecutionScorer() as scorer:
        assert make_execution_reward(scorer).__name__ == "execution_reward"


def test_mismatched_column_lengths_fail_loudly(db: Path) -> None:
    """A misaligned dataset column would silently score every rollout against
    the wrong gold query, which is worse than crashing, because it trains."""
    with ExecutionScorer() as scorer:
        reward = make_execution_reward(scorer)
        with pytest.raises(ValueError, match="aligned"):
            reward(
                prompts=["q", "q"],
                completions=[BLOCK + "SELECT id FROM t"] * 2,
                completion_ids=[[0], [0]],
                gold_sql=[GOLD],  # one short
                db_path=[str(db), str(db)],
            )


def test_exact_match_is_reported_separately_from_the_blended_reward(db: Path) -> None:
    """Mean reward mixes 1.0 matches with 0.1 grounding bonuses, so it cannot
    distinguish 'the policy is sharpening' from 'the policy is farming the
    bonus'. With beta=0 there is no KL anchor holding it back, so that
    distinction is the difference between a useful run and a hacked one. This
    second function is logged by TRL under its own name and carries weight 0,
    so it observes without influencing the gradient."""
    from sqlpup.rl.reward import make_exact_match_reward

    with ExecutionScorer() as scorer:
        observe = make_exact_match_reward(scorer)
        out = observe(
            prompts=["q"] * 3,
            completions=[
                BLOCK + "SELECT id FROM t",  # exact match
                BLOCK + "SELECT name FROM t",  # grounded but wrong -> NOT a match
                BLOCK + "SELECT 1",  # degenerate
            ],
            completion_ids=[[0]] * 3,
            gold_sql=[GOLD] * 3,
            db_path=[str(db)] * 3,
        )
    assert out == [1.0, 0.0, 0.0]
    assert observe.__name__ == "rollout_ex"
