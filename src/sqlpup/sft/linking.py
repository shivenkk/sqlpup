"""Derive schema-linking blocks from gold SQL -- the v3 target prefix.

The block names exactly the schema objects the query touches::

    -- tables: customers, orders
    -- columns: customers.id, customers.name, orders.customer_id

Training targets carry it in front of the SQL so the model must commit to a
binding before writing the query (v2e measured linking F1 0.978 on correct
answers vs 0.667 on wrong ones -- binding *is* the difference). Derivation is
matching, not full parsing, but it is alias-aware: tables come from FROM/JOIN
clauses, ``alias.column`` references resolve through the alias to one table,
and only unqualified leftovers fall back to name matching -- so ``T2.id``
never claims every mentioned table's ``id``, and the block can never teach an
identifier the schema doesn't have.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

_IDENT: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUOTED: Final = re.compile(r'[`"\[]([^`"\]]+)[`"\]]')
_TABLE_REF: Final = re.compile(
    r"\b(?:FROM|JOIN)\s+[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)[`\"\]]?"
    r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)
_QUAL_REF: Final = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
    r"(?:[`\"\[]([^`\"\]]+)[`\"\]]|([A-Za-z_][A-Za-z0-9_]*))"
)
# Words a FROM/JOIN clause can run into that must never be read as an alias.
_NOT_ALIAS: Final = frozenset(
    [
        "where",
        "on",
        "inner",
        "outer",
        "left",
        "right",
        "full",
        "cross",
        "join",
        "as",
        "group",
        "order",
        "having",
        "limit",
        "union",
        "except",
        "intersect",
        "and",
        "or",
        "set",
        "values",
    ]
)


def schema_identifiers(db_path: Path | str) -> dict[str, list[str]]:
    """Table -> ordered column names, in the schema's own casing."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        tables = [
            name
            for (name,) in con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            table: [row[1] for row in con.execute(f'PRAGMA table_info("{table}")')]
            for table in tables
        }
    finally:
        con.close()


def sql_identifier_tokens(sql: str) -> set[str]:
    """Lowercased identifier candidates in *sql*: bare words plus quoted names."""
    tokens = {t.lower() for t in _IDENT.findall(sql)}
    tokens |= {t.lower() for t in _QUOTED.findall(sql)}
    return tokens


def _table_mentions(
    sql: str, schema: Mapping[str, Sequence[str]]
) -> tuple[dict[str, None], dict[str, str]]:
    """FROM/JOIN-mentioned tables (insertion-ordered) and the alias map."""
    by_lower = {table.lower(): table for table in schema}
    mentioned: dict[str, None] = {}
    alias_to_table: dict[str, str] = {}
    for match in _TABLE_REF.finditer(sql):
        table = by_lower.get(match.group(1).lower())
        if table is None:
            continue
        mentioned[table] = None
        alias_to_table[table.lower()] = table
        alias = match.group(2)
        if alias and alias.lower() not in _NOT_ALIAS:
            alias_to_table[alias.lower()] = table
    return mentioned, alias_to_table


def mentioned_tables(sql: str, schema: Mapping[str, Sequence[str]]) -> list[str]:
    """Schema tables the query's FROM/JOIN clauses reference, in order seen."""
    mentioned, _ = _table_mentions(sql, schema)
    return list(mentioned)


def linked_target(sql: str, schema: Mapping[str, Sequence[str]]) -> str:
    """The ``-- tables:/-- columns:`` block for *sql* against *schema*."""
    mentioned, alias_to_table = _table_mentions(sql, schema)

    claimed: set[tuple[str, str]] = set()
    unqualified = sql
    for match in _QUAL_REF.finditer(sql):
        table = alias_to_table.get(match.group(1).lower())
        column_name = (match.group(2) or match.group(3)).lower()
        if table is not None:
            for column in schema[table]:
                if column.lower() == column_name:
                    claimed.add((table, column))
        # Qualified references are resolved; keep them out of the fallback pool.
        unqualified = unqualified.replace(match.group(0), " ")

    tokens = sql_identifier_tokens(unqualified)
    for table in mentioned:
        for column in schema[table]:
            if column.lower() in tokens:
                claimed.add((table, column))

    tables = sorted(mentioned, key=str.lower)
    columns = [
        f"{table}.{column}"
        for table in tables
        for column in schema[table]
        if (table, column) in claimed
    ]
    tables_line = ", ".join(tables)
    columns_line = ", ".join(columns)
    return (
        f"-- tables:{' ' if tables_line else ''}{tables_line}\n"
        f"-- columns:{' ' if columns_line else ''}{columns_line}\n"
    )
