"""GRPO execution reward: match=1.0, executes-but-wrong=0.1, broken=0.0."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlpup.eval import ExecutionScorer
from sqlpup.rl.reward import EXECUTES_BONUS, execution_reward


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE t (id INTEGER, name TEXT);
        INSERT INTO t (id, name) VALUES (1, 'alpha'), (2, 'beta');
        """
    )
    con.commit()
    con.close()
    return path


GOLD = "SELECT id FROM t ORDER BY id"


def test_matching_prediction_earns_full_reward(fixture_db: Path) -> None:
    with ExecutionScorer() as scorer:
        assert execution_reward("SELECT id FROM t", GOLD, fixture_db, scorer) == 1.0


def test_executable_but_wrong_earns_the_format_bonus(fixture_db: Path) -> None:
    with ExecutionScorer() as scorer:
        reward = execution_reward("SELECT name FROM t", GOLD, fixture_db, scorer)
    assert reward == EXECUTES_BONUS
    assert 0.0 < reward < 1.0


def test_broken_sql_earns_zero(fixture_db: Path) -> None:
    with ExecutionScorer() as scorer:
        assert execution_reward("SELEC id FROM t", GOLD, fixture_db, scorer) == 0.0


def test_empty_sql_earns_zero(fixture_db: Path) -> None:
    with ExecutionScorer() as scorer:
        assert execution_reward("", GOLD, fixture_db, scorer) == 0.0


def test_write_attempt_earns_zero(fixture_db: Path) -> None:
    with ExecutionScorer() as scorer:
        assert execution_reward("DELETE FROM t", GOLD, fixture_db, scorer) == 0.0


# --- reward hacking: the bonus must require touching the database ------------
#
# Measured before the guard existed: SELECT 1, SELECT 'x' and SELECT 1 AS a all
# earned the full 0.1 executes-bonus. Under group-relative advantage that is a
# degenerate attractor -- a constant, trivially-learnable string that always
# beats broken SQL, costs almost no tokens, and needs no schema understanding.
# A 394M policy would find it quickly, and the collapse would look like
# "training is working" (reward up, length down) until evaluation.


def test_a_query_touching_no_table_earns_nothing(fixture_db: Path) -> None:
    with ExecutionScorer() as scorer:
        for degenerate in ("SELECT 1", "SELECT 'x'", "SELECT 1 AS a", "SELECT 1 WHERE 0"):
            assert execution_reward(degenerate, GOLD, fixture_db, scorer) == 0.0, degenerate


def test_grounded_but_wrong_still_earns_the_bonus(fixture_db: Path) -> None:
    """The dense signal must survive the guard: referencing a real table is the
    behaviour we want to reinforce early, even when the answer is wrong."""
    with ExecutionScorer() as scorer:
        assert execution_reward("SELECT name FROM t", GOLD, fixture_db, scorer) == EXECUTES_BONUS


def test_a_correct_answer_is_never_penalised_by_the_guard(fixture_db: Path) -> None:
    """A gold query that happens to select a constant must still score 1.0 --
    the guard gates the *bonus*, never a match."""
    with ExecutionScorer() as scorer:
        assert execution_reward("SELECT 1", "SELECT 1", fixture_db, scorer) == 1.0
