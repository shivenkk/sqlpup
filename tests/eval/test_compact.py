"""Gold-blind schema compaction for prompts that overflow the context.

Measured problem: ``_plan_batches`` drops every prompt at or over the context
limit and records ``""`` for it. On the development set that is a defensible
scoring choice (an empty prediction is a miss, so accuracy is never inflated by
skipping the hardest schemas). On a *submission* it is not: BIRD treats an
empty prediction as an abnormal output and flags a run whose abnormal rate
exceeds 5%.

The fallback here is gold-blind by construction: it never reads the reference
query, only the database catalog. Each level keeps every table and every column
*name* reachable -- it removes verbosity, not schema -- until the last level,
which caps columns per table but always keeps keys, so joins stay expressible.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlpup.eval.compact import compact_ddl, fit_ddl


def _db(tmp_path: Path, statements: list[str], name: str = "t.sqlite") -> Path:
    path = tmp_path / name
    con = sqlite3.connect(path)
    try:
        for statement in statements:
            con.execute(statement)
        con.commit()
    finally:
        con.close()
    return path


WIDE = [
    """CREATE TABLE "school district" (
        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        name VARCHAR(255) DEFAULT NULL,
        county TEXT DEFAULT 'unknown' NOT NULL,
        funding REAL DEFAULT 0.0,
        notes TEXT
    )""",
    """CREATE TABLE pupil (
        pupil_id INTEGER PRIMARY KEY,
        district_id INTEGER REFERENCES "school district"(id),
        score REAL DEFAULT NULL,
        comment TEXT DEFAULT NULL
    )""",
]


def test_level0_is_the_verbatim_schema(tmp_path: Path) -> None:
    """Level 0 must be a no-op, so the fallback ladder starts where the
    ordinary prompt starts and any difference is attributable to compaction."""
    from sqlpup.eval.prompts import schema_ddl

    db = _db(tmp_path, WIDE)
    assert compact_ddl(db, level=0) == schema_ddl(db)


def test_level1_drops_decoration_but_keeps_every_identifier(tmp_path: Path) -> None:
    """Defaults, AUTOINCREMENT and NOT NULL tell the model nothing about which
    column answers a question, but they cost context on every wide table."""
    db = _db(tmp_path, WIDE)
    out = compact_ddl(db, level=1)
    for identifier in ("school district", "pupil", "county", "funding", "notes", "score"):
        assert identifier in out, identifier
    assert "AUTOINCREMENT" not in out.upper()
    assert "DEFAULT" not in out.upper()
    assert len(out) < len(compact_ddl(db, level=0))


def test_level1_preserves_quoting_for_names_that_need_it(tmp_path: Path) -> None:
    """``school district`` has a space: unquoted it is a syntax error, and the
    prompt is supposed to show the model executable DDL."""
    db = _db(tmp_path, WIDE)
    assert '"school district"' in compact_ddl(db, level=1)


def test_level2_keeps_keys_when_it_caps_columns(tmp_path: Path) -> None:
    """Dropping a foreign key would make a join inexpressible, which is a worse
    failure than a long prompt: the model could not answer even in principle."""
    db = _db(tmp_path, WIDE)
    out = compact_ddl(db, level=2, max_columns=2)
    assert "pupil_id" in out  # primary key survives the cap
    assert "district_id" in out  # foreign key survives the cap
    assert len(out) < len(compact_ddl(db, level=1))


def test_compaction_output_is_still_executable_sql(tmp_path: Path) -> None:
    """Every level must produce DDL SQLite accepts; a malformed schema is a
    worse training/prompting signal than a truncated one (the same rule the
    training-side reducer follows)."""
    db = _db(tmp_path, WIDE)
    for level in (0, 1, 2):
        con = sqlite3.connect(":memory:")
        try:
            con.executescript(compact_ddl(db, level=level, max_columns=2))
        finally:
            con.close()


def test_fit_ddl_returns_the_first_level_that_fits(tmp_path: Path) -> None:
    """The ladder stops as soon as the prompt fits: we spend the minimum
    compaction needed, so most databases are unaffected."""
    db = _db(tmp_path, WIDE)
    full = compact_ddl(db, level=0)

    ddl, level = fit_ddl(db, budget=len(full) + 10, measure=len)
    assert level == 0 and ddl == full

    ddl, level = fit_ddl(db, budget=len(compact_ddl(db, level=1)) + 5, measure=len)
    assert level == 1


def test_fit_ddl_returns_its_smallest_form_when_nothing_fits(tmp_path: Path) -> None:
    """A budget no level can meet must still yield a usable prompt. Returning
    the most compact form and letting the caller proceed is what converts a
    guaranteed empty prediction into a possible answer -- the whole point."""
    db = _db(tmp_path, WIDE)
    ddl, level = fit_ddl(db, budget=1, measure=len)
    assert level == 2
    assert "pupil" in ddl and ddl.strip()


def test_fit_ddl_never_consults_the_reference_query(tmp_path: Path) -> None:
    """Gold-blindness is the property that makes this legitimate at submission
    time; it is enforced by signature (there is nowhere to pass gold SQL)."""
    import inspect

    parameters = set(inspect.signature(fit_ddl).parameters)
    assert not parameters & {"sql", "gold", "gold_sql", "query"}


def test_measure_can_count_tokens_rather_than_characters(tmp_path: Path) -> None:
    """The real budget is tokens; characters are only the test's stand-in."""
    db = _db(tmp_path, WIDE)
    calls: list[str] = []

    def fake_tokens(text: str) -> int:
        calls.append(text)
        return len(text) // 4

    _, level = fit_ddl(db, budget=10**6, measure=fake_tokens)
    assert level == 0
    assert calls, "measure must actually be consulted"


def test_empty_database_does_not_crash_the_fallback(tmp_path: Path) -> None:
    db = _db(tmp_path, ["CREATE TABLE only (a INTEGER)"])
    con = sqlite3.connect(db)
    con.execute("DROP TABLE only")
    con.commit()
    con.close()
    ddl, _ = fit_ddl(db, budget=1, measure=len)
    assert ddl == ""


@pytest.mark.parametrize("level", [-1, 3])
def test_unknown_level_is_rejected(tmp_path: Path, level: int) -> None:
    db = _db(tmp_path, WIDE)
    with pytest.raises(ValueError):
        compact_ddl(db, level=level)
