"""Refine loop: probe errors feed repair prompts; both EX framings stay reportable."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlpup.eval import ExecutionScorer
from sqlpup.eval.generate import FakeGenerator
from sqlpup.eval.refine import refine_batch


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


def test_clean_first_attempt_needs_no_retry(fixture_db: Path) -> None:
    fake = FakeGenerator(script=[["SELECT id FROM t;"]])
    with ExecutionScorer() as scorer:
        results = refine_batch(["P\n-- SQL:\n"], [fixture_db], fake, scorer)
    (result,) = results
    assert result.used_retries == 0
    assert result.final_sql == "SELECT id FROM t"
    assert result.single_shot_sql == "SELECT id FROM t"
    assert len(fake.calls) == 1


def test_error_feeds_repair_prompt_and_gets_fixed(fixture_db: Path) -> None:
    fake = FakeGenerator(script=[["SELECT missing_col FROM t"], ["SELECT id FROM t"]])
    with ExecutionScorer() as scorer:
        results = refine_batch(["P\n-- SQL:\n"], [fixture_db], fake, scorer)
    (result,) = results
    assert result.used_retries == 1
    assert not result.attempts[0].executed_ok
    assert result.attempts[1].executed_ok
    assert result.final_sql == "SELECT id FROM t"
    assert result.single_shot_sql == "SELECT missing_col FROM t"
    repair_prompt = fake.calls[1][0]
    assert "SELECT missing_col FROM t" in repair_prompt
    assert "missing_col" in repair_prompt  # the error text itself
    assert repair_prompt.endswith("-- Corrected SQL:\n")


def test_identical_retry_short_circuits(fixture_db: Path) -> None:
    fake = FakeGenerator(script=[["SELEC 1"], ["SELEC 1"]])
    with ExecutionScorer() as scorer:
        results = refine_batch(["P\n-- SQL:\n"], [fixture_db], fake, scorer, max_retries=2)
    (result,) = results
    # The identical round-1 output is dropped, not appended, and round 2 never runs.
    assert len(result.attempts) == 1
    assert len(fake.calls) == 2


def test_retries_are_capped(fixture_db: Path) -> None:
    fake = FakeGenerator(script=[["SELEC 1"], ["SELEC 2"], ["SELEC 3"]])
    with ExecutionScorer() as scorer:
        results = refine_batch(["P\n-- SQL:\n"], [fixture_db], fake, scorer, max_retries=2)
    (result,) = results
    assert result.used_retries == 2
    assert not result.attempts[-1].executed_ok
    assert len(fake.calls) == 3


def test_only_failing_examples_are_reprompted(fixture_db: Path) -> None:
    fake = FakeGenerator(
        script=[["SELECT id FROM t", "SELECT missing_col FROM t"], ["SELECT name FROM t"]]
    )
    with ExecutionScorer() as scorer:
        results = refine_batch(
            ["P0\n-- SQL:\n", "P1\n-- SQL:\n"], [fixture_db, fixture_db], fake, scorer
        )
    assert len(fake.calls[0]) == 2
    assert len(fake.calls[1]) == 1  # only the failing example came back
    assert results[0].used_retries == 0
    assert results[1].used_retries == 1
    assert results[1].final_sql == "SELECT name FROM t"


def test_mismatched_lengths_are_rejected(fixture_db: Path) -> None:
    fake = FakeGenerator(script=[])
    with ExecutionScorer() as scorer, pytest.raises(ValueError):
        refine_batch(["only-prompt"], [fixture_db, fixture_db], fake, scorer)
