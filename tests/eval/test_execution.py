"""Single-example execution-match core (the GRPO-reward-reusable unit).

Every test builds a tiny SQLite database in-process -- no network, no real BIRD
databases. The behaviours asserted mirror the official BIRD evaluator's
``set(predicted_rows) == set(gold_rows)`` rule plus the sandbox's safety and
resource bounds.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from sqlpup.eval import ExecutionScorer, MatchCategory, MatchResult, execution_match

# A recursive CTE with no termination: it never returns, so only an OS-level
# kill of the worker can stop it (the whole point of the process sandbox).
RUNAWAY_SQL = (
    "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT COUNT(*) FROM r"
)


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    """A small database: ``t`` (with a duplicate name) and a 5000-row ``nums``."""
    path = tmp_path / "fixture.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE t (id INTEGER, name TEXT);
        INSERT INTO t (id, name) VALUES (1, 'alpha'), (2, 'beta'), (3, 'alpha');
        CREATE TABLE nums (n INTEGER);
        """
    )
    con.executemany("INSERT INTO nums (n) VALUES (?)", [(i,) for i in range(5000)])
    con.commit()
    con.close()
    return path


# --- official set-equality semantics ----------------------------------------


def test_exact_match(fixture_db: Path) -> None:
    result = execution_match("SELECT id FROM t", "SELECT id FROM t", fixture_db)
    assert result == MatchResult(match=True, category=MatchCategory.OK)


def test_order_insensitive_match(fixture_db: Path) -> None:
    # Same rows, opposite order -> the unordered set comparison still matches.
    result = execution_match(
        "SELECT id FROM t ORDER BY id DESC",
        "SELECT id FROM t ORDER BY id ASC",
        fixture_db,
    )
    assert result.match is True
    assert result.category is MatchCategory.OK


def test_duplicate_rows_collapse_under_set(fixture_db: Path) -> None:
    # gold returns 'alpha' twice; DISTINCT pred returns it once. set() collapses
    # duplicates on both sides, so the official rule counts this a match.
    result = execution_match(
        "SELECT DISTINCT name FROM t",
        "SELECT name FROM t",
        fixture_db,
    )
    assert result.match is True


def test_genuine_mismatch(fixture_db: Path) -> None:
    result = execution_match("SELECT id FROM t WHERE id = 1", "SELECT id FROM t", fixture_db)
    assert result.match is False
    assert result.category is MatchCategory.OK


def test_both_empty_is_a_match(fixture_db: Path) -> None:
    # Official semantics: two empty result sets compare equal -> match.
    result = execution_match(
        "SELECT id FROM t WHERE 1 = 0",
        "SELECT id FROM t WHERE 0 = 1",
        fixture_db,
    )
    assert result.match is True
    assert result.category is MatchCategory.EMPTY


def test_empty_prediction_against_nonempty_gold_mismatches(fixture_db: Path) -> None:
    result = execution_match("SELECT id FROM t WHERE 1 = 0", "SELECT id FROM t", fixture_db)
    assert result.match is False
    assert result.category is MatchCategory.OK


# --- failure handling: never crash or hang ----------------------------------


def test_predicted_sql_error_is_non_match_not_crash(fixture_db: Path) -> None:
    result = execution_match("SELECT * FROM does_not_exist", "SELECT id FROM t", fixture_db)
    assert result.match is False
    assert result.category is MatchCategory.EXECUTION_ERROR


def test_gold_sql_error_is_loud_and_distinct(fixture_db: Path) -> None:
    # A broken gold is a harness problem, tagged distinctly so it is never
    # silently scored as an ordinary wrong answer.
    result = execution_match("SELECT id FROM t", "SELECT * FROM does_not_exist", fixture_db)
    assert result.match is False
    assert result.category is MatchCategory.GOLD_ERROR


def test_timeout_is_bounded_and_non_match(fixture_db: Path) -> None:
    # A runaway recursive CTE never returns; the killable worker must interrupt
    # it and report a timeout in bounded wall-clock (a signal-based timeout is
    # what froze the prior attempt on macOS EINTR).
    start = time.monotonic()
    result = execution_match(RUNAWAY_SQL, "SELECT id FROM t", fixture_db, timeout=1.0)
    elapsed = time.monotonic() - start
    assert result.match is False
    assert result.category is MatchCategory.TIMEOUT
    assert elapsed < 20.0  # proven interrupted, not run to completion


# --- read-only sandbox: no writes to the database, ever ----------------------


def _row_count(db_path: Path, table: str) -> int:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        con.close()


def test_predicted_insert_cannot_mutate_db(fixture_db: Path) -> None:
    before = _row_count(fixture_db, "t")
    result = execution_match(
        "INSERT INTO t (id, name) VALUES (99, 'x')", "SELECT id FROM t", fixture_db
    )
    assert result.match is False
    assert result.category is MatchCategory.EXECUTION_ERROR
    assert _row_count(fixture_db, "t") == before  # unchanged


def test_predicted_drop_cannot_mutate_db(fixture_db: Path) -> None:
    result = execution_match("DROP TABLE t", "SELECT id FROM t", fixture_db)
    assert result.match is False
    assert result.category is MatchCategory.EXECUTION_ERROR
    assert _row_count(fixture_db, "t") == 3  # table still present with its rows


def test_predicted_update_cannot_mutate_db(fixture_db: Path) -> None:
    result = execution_match("UPDATE t SET name = 'zzz'", "SELECT name FROM t", fixture_db)
    assert result.match is False
    assert result.category is MatchCategory.EXECUTION_ERROR
    con = sqlite3.connect(f"file:{fixture_db}?mode=ro", uri=True)
    names = {row[0] for row in con.execute("SELECT name FROM t")}
    con.close()
    assert names == {"alpha", "beta"}  # nothing was rewritten


def test_predicted_attach_cannot_create_files(fixture_db: Path, tmp_path: Path) -> None:
    # ATTACH is the escape that read-only mode alone does not close; the
    # authorizer must reject it so no new file appears on disk.
    target = tmp_path / "escape.db"
    result = execution_match(f"ATTACH DATABASE '{target}' AS evil", "SELECT id FROM t", fixture_db)
    assert result.match is False
    assert result.category is MatchCategory.EXECUTION_ERROR
    assert not target.exists()


# --- memory bound: distinct-row cap ------------------------------------------


def test_prediction_exceeding_row_limit_is_non_match(fixture_db: Path) -> None:
    # gold stays under the cap; the prediction's distinct rows blow past it, so
    # the sets cannot be equal -> non-match tagged row_limit.
    result = execution_match(
        "SELECT n FROM nums",
        "SELECT n FROM nums WHERE n < 10",
        fixture_db,
        row_limit=100,
    )
    assert result.match is False
    assert result.category is MatchCategory.ROW_LIMIT


def test_gold_exceeding_row_limit_is_loud(fixture_db: Path) -> None:
    result = execution_match(
        "SELECT n FROM nums WHERE n < 10",
        "SELECT n FROM nums",
        fixture_db,
        row_limit=100,
    )
    assert result.match is False
    assert result.category is MatchCategory.GOLD_ERROR


# --- reusable scorer: one worker across many pairs ---------------------------


def test_scorer_reuses_one_worker_and_recovers_after_timeout(fixture_db: Path) -> None:
    with ExecutionScorer(timeout=1.0) as scorer:
        first = scorer.score("SELECT id FROM t", "SELECT id FROM t", fixture_db)
        assert first.match is True
        # a timeout kills the worker; the scorer must respawn it transparently
        timed_out = scorer.score(RUNAWAY_SQL, "SELECT id FROM t", fixture_db)
        assert timed_out.category is MatchCategory.TIMEOUT
        # subsequent scoring still works on the respawned worker
        after = scorer.score(
            "SELECT id FROM t WHERE id = 1", "SELECT id FROM t WHERE id = 1", fixture_db
        )
        assert after.match is True


def test_execution_match_returns_typed_result(fixture_db: Path) -> None:
    # The reward-reuse contract: importable, pure (pred, gold, db) -> typed result.
    result = execution_match("SELECT 1", "SELECT 1", fixture_db)
    assert isinstance(result, MatchResult)
    assert isinstance(result.match, bool)
    assert isinstance(result.category, MatchCategory)
