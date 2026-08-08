"""Two-pass generation: the model narrows its own schema, then answers.

Pass 1 is an ordinary generation; its linking block is a *prediction of which
tables matter*. Pass 2 re-asks the same question with only those tables' DDL,
so the model chooses columns inside a handful of tables rather than forty.

Why this shape (measured on v3 step-15K, 500 Mini-Dev examples): of the
binding failures in the largest error pool, 40.9% had **all** gold tables
already named in the model's own block and a further 37.4% had some, while
only 2.6% named none of them. The table decision is mostly right; the column
decision inside it is what fails. Narrowing is therefore free information the
model already produced, and it buys the context budget to add BIRD's column
descriptions -- which are unaffordable against a full schema.

Everything here is **gold-blind**: it reads the model's output and the
database, never the reference SQL. Narrowing that matches nothing falls back
to the full schema, so a bad pass-1 block can waste a pass but cannot blind
the model.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from sqlpup.eval.prompts import schema_ddl

_TABLES_LINE: Final = re.compile(r"^--\s*tables:\s*(.*)$", re.IGNORECASE)
# Descriptions are a hint, not the schema: keep them well inside the budget
# the narrowed DDL just freed up.
DEFAULT_NOTE_LINES: Final = 12


def tables_from_block(raw_completion: str) -> list[str]:
    """Table names the completion's linking block claims, in order."""
    for line in raw_completion.splitlines():
        match = _TABLES_LINE.match(line.strip())
        if match is None:
            if line.strip().startswith("--"):
                continue  # a columns line before a tables line: keep looking
            break  # the block is over
        return [name.strip() for name in match.group(1).split(",") if name.strip()]
    return []


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _adds_information(column: str, human: str, description: str) -> bool:
    """Does this gloss say anything the column name does not already say?

    BIRD's description files are mostly tautological (`City: City`); emitting
    them spends context on nothing, which is how the unfiltered version cost
    4.2 EX points at step-30K.
    """
    key = _normalise(column)
    return any(text and _normalise(text) != key for text in (human, description))


def _create_statements(db_path: Path | str) -> dict[str, str]:
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        return {
            name: sql
            for name, sql in con.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
                "ORDER BY name"
            )
        }
    finally:
        con.close()


def _fk_neighbours(db_path: Path | str, tables: Sequence[str]) -> list[str]:
    """Degree-1 foreign-key parents and children of *tables*."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        all_tables = [
            name
            for (name,) in con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        wanted = {t.lower() for t in tables}
        extra: list[str] = []
        for name in all_tables:
            parents = {row[2].lower() for row in con.execute(f'PRAGMA foreign_key_list("{name}")')}
            if name.lower() in wanted:
                extra.extend(p for p in parents)  # parents of a named table
            elif parents & wanted:
                extra.append(name)  # children pointing at one
        by_lower = {t.lower(): t for t in all_tables}
        return [by_lower[e.lower()] for e in extra if e.lower() in by_lower]
    finally:
        con.close()


def focused_schema(db_path: Path | str, tables: Iterable[str], *, expand_fk: bool = False) -> str:
    """DDL for just *tables*; the full schema when none of them are real.

    ``expand_fk`` also keeps their degree-1 foreign-key neighbours, so a block
    that named only some of the tables a query needs can still reach the rest
    through a join path.
    """
    statements = _create_statements(db_path)
    by_lower = {name.lower(): name for name in statements}
    named = list(tables)
    wanted = [by_lower[t.lower()] for t in named if t.lower() in by_lower]
    if not wanted:
        return schema_ddl(db_path)
    if expand_fk:
        wanted = wanted + _fk_neighbours(db_path, wanted)
    seen: dict[str, None] = {}
    for name in wanted:  # preserve the model's order, drop duplicates
        seen[name] = None
    return "\n\n".join(f"{statements[name]};" for name in seen)


def column_notes(
    db_path: Path | str,
    tables: Sequence[str],
    *,
    max_lines: int = DEFAULT_NOTE_LINES,
) -> str:
    """BIRD's human-readable column names for *tables*, as comment lines.

    BIRD ships ``database_description/<table>.csv`` mapping cryptic column
    names to human ones (``NCESDist`` -> "National Center for Educational
    Statistics school district identification number") -- exactly the bridge
    for a question that names a concept the schema spells differently. Absent
    files yield an empty string; these are a hint layer, never required.
    """
    root = Path(db_path).parent / "database_description"
    if not root.is_dir():
        return ""
    statements = _create_statements(db_path)
    by_lower = {name.lower(): name for name in statements}
    lines: list[str] = []
    for table in tables:
        real = by_lower.get(table.lower())
        if real is None:
            continue
        path = root / f"{real}.csv"
        if not path.exists():
            continue
        # BIRD's description files are inconsistently encoded; never let one
        # bad byte cost us the whole hint layer.
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                if len(lines) >= max_lines:
                    return "\n".join(lines)
                column = (row.get("original_column_name") or "").strip()
                human = (row.get("column_name") or "").strip()
                description = (row.get("column_description") or "").strip()
                if not column or (not human and not description):
                    continue
                if not _adds_information(column, human, description):
                    continue
                gloss = human or description
                if human and description and description.lower() != human.lower():
                    gloss = f"{human} -- {description}"
                lines.append(f"-- {real}.{column}: {gloss[:120]}")
    return "\n".join(lines)
