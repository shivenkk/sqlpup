"""Execution-guided self-consistency: sample k, keep what most of them agree on.

Motivated by a measured result: voting across four existing
prediction sets scored 15.2% against 12.6% for the best member, and the oracle
union over those members reached 20.4% -- the correct answer is often
reachable but not reliably produced. This module is the single-model form of
that idea, which is what a shippable system can actually use.

Candidates are grouped by the *result set they produce*, not by their text:
two different queries that return the same rows are the same answer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlpup.eval.execution import ExecutionScorer
from sqlpup.eval.vote import majority_vote


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "toy.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE t (id INTEGER, name TEXT);INSERT INTO t VALUES (1,'a'),(2,'b');"
    )
    con.commit()
    con.close()
    return path


def test_picks_the_answer_most_candidates_agree_on(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with ExecutionScorer() as scorer:
        winner, stats = majority_vote(
            [
                "SELECT name FROM t ORDER BY id",  # agrees with #2 on rows
                "SELECT name FROM t",  # same rows, different text
                "SELECT id FROM t",  # a different answer
            ],
            db,
            scorer,
        )
    assert winner in ("SELECT name FROM t ORDER BY id", "SELECT name FROM t")
    assert stats["votes"] == 2
    assert stats["valid"] == 3
    assert stats["distinct_answers"] == 2


def test_invalid_candidates_do_not_vote(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with ExecutionScorer() as scorer:
        winner, stats = majority_vote(
            ["SELECT nope FROM t", "SELECT syntax error(", "SELECT id FROM t"], db, scorer
        )
    assert winner == "SELECT id FROM t"
    assert stats["valid"] == 1
    assert stats["votes"] == 1


def test_falls_back_to_the_first_candidate_when_none_execute(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with ExecutionScorer() as scorer:
        winner, stats = majority_vote(["SELECT nope FROM t", "ALSO BROKEN"], db, scorer)
    assert winner == "SELECT nope FROM t"  # something must be submitted
    assert stats["valid"] == 0


def test_ties_prefer_the_earlier_candidate(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with ExecutionScorer() as scorer:
        winner, _ = majority_vote(["SELECT id FROM t", "SELECT name FROM t"], db, scorer)
    assert winner == "SELECT id FROM t"  # greedy sample is candidate 0 by convention


def test_empty_results_lose_to_any_non_empty_answer(tmp_path: Path) -> None:
    """Measured over BIRD Mini-Dev (n=500): zero gold queries return an empty
    result, yet empty clusters won 20% of votes -- always wrongly, because
    wrong queries agree on emptiness while correct ones return distinct rows.
    Demoting empties took cross-checkpoint voting from 15.2% to 16.4%."""
    db = _db(tmp_path)
    with ExecutionScorer() as scorer:
        winner, stats = majority_vote(
            [
                "SELECT name FROM t WHERE id = 999",  # empty
                "SELECT name FROM t WHERE id = 998",  # empty, same fingerprint
                "SELECT name FROM t",  # real rows, outvoted 2-1
            ],
            db,
            scorer,
        )
    assert winner == "SELECT name FROM t"
    assert stats["empty_demoted"] is True


def test_empty_still_wins_when_nothing_else_executes(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with ExecutionScorer() as scorer:
        winner, stats = majority_vote(
            ["SELECT broken(", "SELECT name FROM t WHERE id = 999"], db, scorer
        )
    assert winner == "SELECT name FROM t WHERE id = 999"  # better than nothing
    assert stats["empty_demoted"] is False
