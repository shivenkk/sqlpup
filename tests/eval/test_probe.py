"""The probe op: execute one SQL alone and surface its error text (refine signal)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from sqlpup.eval import ExecutionScorer

RUNAWAY_SQL = (
    "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT COUNT(*) FROM r"
)


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


def test_probe_ok_returns_true_and_empty_error(fixture_db: Path) -> None:
    with ExecutionScorer() as scorer:
        ok, error = scorer.probe("SELECT id FROM t", fixture_db)
    assert (ok, error) == (True, "")


def test_probe_syntax_error_surfaces_the_message(fixture_db: Path) -> None:
    with ExecutionScorer() as scorer:
        ok, error = scorer.probe("SELEC id FROM t", fixture_db)
    assert not ok
    assert "syntax error" in error


def test_probe_unknown_column_surfaces_the_message(fixture_db: Path) -> None:
    with ExecutionScorer() as scorer:
        ok, error = scorer.probe("SELECT missing_col FROM t", fixture_db)
    assert not ok
    assert "missing_col" in error


def test_probe_write_attempt_is_an_error(fixture_db: Path) -> None:
    with ExecutionScorer() as scorer:
        ok, error = scorer.probe("INSERT INTO t (id, name) VALUES (9, 'x')", fixture_db)
    assert not ok
    assert error  # denied by the read-only guards, with a message


def test_probe_row_limit_overflow_still_counts_as_ok(fixture_db: Path) -> None:
    # The cap bounds worker memory; a huge-but-valid result still *executes*.
    with ExecutionScorer(row_limit=1) as scorer:
        ok, error = scorer.probe("SELECT id FROM t", fixture_db)
    assert (ok, error) == (True, "")


def test_probe_timeout_is_bounded_and_worker_recovers(fixture_db: Path) -> None:
    with ExecutionScorer(timeout=0.5) as scorer:
        start = time.monotonic()
        ok, error = scorer.probe(RUNAWAY_SQL, fixture_db)
        elapsed = time.monotonic() - start
        assert not ok
        assert "timeout" in error
        assert elapsed < 5.0
        # The killed worker respawns transparently for the next probe.
        assert scorer.probe("SELECT 1", fixture_db) == (True, "")
