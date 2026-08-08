"""Gold-blind schema compaction for prompts that overflow the context.

A prompt at or over the context limit is not generated for at all: the planner
sets it aside and the prediction stays ``""``. On the development set that is
the honest choice -- an empty prediction scores as a miss, so accuracy is never
inflated by quietly skipping the largest schemas. At submission time it is the
wrong choice, because BIRD counts an empty prediction as an abnormal output and
flags a run whose abnormal rate exceeds 5%.

This module supplies the fallback. It reads only the database catalog, never
the reference query, so it is legitimate on a test set whose answers we do not
have. The ladder spends the least compaction that fits:

* **level 0** -- the verbatim schema, identical to the ordinary prompt.
* **level 1** -- tables rebuilt from catalog metadata with every table and every
  column name intact, minus the decoration (``DEFAULT``, ``NOT NULL``,
  ``AUTOINCREMENT``) that costs context on wide tables and tells the model
  nothing about which column answers a question.
* **level 2** -- additionally caps columns per table, always keeping primary and
  foreign keys, so every join the full schema permitted stays expressible.

Like the training-side reducer, output is executable SQL reconstructed from
metadata rather than string-sliced text: a malformed schema is a worse prompt
than a compact one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Final

from sqlpup.eval.prompts import schema_ddl

MAX_LEVEL: Final = 2
DEFAULT_MAX_COLUMNS: Final = 8


def _quote(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _connect(db_path: Path | str) -> sqlite3.Connection:
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _table_names(con: sqlite3.Connection) -> list[str]:
    return [
        name
        for (name,) in con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _render_table(con: sqlite3.Connection, table: str, *, max_columns: int | None) -> str:
    columns = list(con.execute(f"PRAGMA table_info({_quote(table)})"))
    if not columns:
        return ""
    foreign_keys = list(con.execute(f"PRAGMA foreign_key_list({_quote(table)})"))

    # Keys are structural: dropping one makes a join inexpressible, which is a
    # worse failure than a long prompt because the model could not answer even
    # in principle. They are kept regardless of the cap.
    # table_info rows are (cid, name, type, notnull, default, pk);
    # foreign_key_list rows are (id, seq, table, from, to, ...).
    key_columns = {str(row[1]) for row in columns if int(row[5]) > 0}
    key_columns |= {str(row[3]) for row in foreign_keys}

    kept: list[tuple[str, str]] = []
    for row in columns:
        name, declared_type = str(row[1]), str(row[2] or "")
        if max_columns is not None and len(kept) >= max_columns and name not in key_columns:
            continue
        kept.append((name, declared_type))
    if max_columns is not None:
        # A cap smaller than the key set still keeps the keys; ordinary columns
        # yield first so the budget is spent on what carries join structure.
        mandatory = [pair for pair in kept if pair[0] in key_columns]
        optional = [pair for pair in kept if pair[0] not in key_columns]
        room = max(0, max_columns - len(mandatory))
        keep_names = {name for name, _ in mandatory} | {name for name, _ in optional[:room]}
        kept = [pair for pair in kept if pair[0] in keep_names]

    lines = [f"  {_quote(name)}{' ' + kind if kind else ''}" for name, kind in kept]

    # pk is an ordinal for composite keys, so sort by it to restore key order.
    primary = [
        str(row[1])
        for row in sorted((r for r in columns if int(r[5]) > 0), key=lambda r: int(r[5]))
    ]
    if primary:
        lines.append("  PRIMARY KEY (" + ", ".join(_quote(c) for c in primary) + ")")
    for row in foreign_keys:
        parent, from_column, to_column = str(row[2]), str(row[3]), row[4]
        target = f"({_quote(str(to_column))})" if to_column is not None else ""
        lines.append(f"  FOREIGN KEY ({_quote(from_column)}) REFERENCES {_quote(parent)}{target}")
    return f"CREATE TABLE {_quote(table)} (\n" + ",\n".join(lines) + "\n);"


def compact_ddl(db_path: Path | str, *, level: int, max_columns: int = DEFAULT_MAX_COLUMNS) -> str:
    """The database's schema at compaction *level* (see module docstring)."""
    if not 0 <= level <= MAX_LEVEL:
        raise ValueError(f"compaction level must be 0..{MAX_LEVEL}, got {level}")
    if level == 0:
        return schema_ddl(db_path)
    cap = max_columns if level == MAX_LEVEL else None
    con = _connect(db_path)
    try:
        rendered = [
            statement
            for table in _table_names(con)
            if (statement := _render_table(con, table, max_columns=cap))
        ]
    finally:
        con.close()
    return "\n\n".join(rendered)


def fit_ddl(
    db_path: Path | str,
    *,
    budget: int,
    measure: Callable[[str], int],
    max_columns: int = DEFAULT_MAX_COLUMNS,
) -> tuple[str, int]:
    """The least-compacted schema fitting *budget*, as ``(ddl, level)``.

    *measure* is the caller's size function -- token count in production,
    where the budget is the context the prompt's other parts leave over. When
    no level fits, the most compact form is returned anyway: a prompt that is
    still too long may yet produce an answer, while an omitted one never can.

    There is deliberately no parameter through which reference SQL could be
    passed; gold-blindness is what makes this legitimate on a test set.
    """
    smallest = ""
    for level in range(MAX_LEVEL + 1):
        smallest = compact_ddl(db_path, level=level, max_columns=max_columns)
        if measure(smallest) <= budget:
            return smallest, level
    return smallest, MAX_LEVEL
