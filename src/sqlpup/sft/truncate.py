"""Gold-aware DDL reduction -- recover oversized pairs, keep JOIN paths.

v2e discarded every pair whose prompt+completion exceeded the 2048 context
(22.5%% of SynSQL, 10.4%% of BIRD) because truncating DDL mid-table would
teach malformed schemas. This module reduces instead of truncates, in two
levels tried in order until the pair fits:

* **Level 1** keeps only the query's tables *plus their degree-1 foreign-key
  neighbours* (DIN-SQL practice: a JOIN path the model should learn must stay
  visible even when the gold query doesn't touch the neighbour), verbatim.
* **Level 2** additionally rebuilds each surviving table with only its
  referenced, primary-key, and foreign-key columns -- keys always survive, so
  the reduced schema still describes every join the full schema allowed.

Reduced DDL is real, executable SQL (level 2 is reconstructed from PRAGMA
metadata, not string-sliced), and reduction is deterministic: name-ordered
tables, schema-ordered columns.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sqlpup.eval.prompts import schema_ddl
from sqlpup.sft.linking import mentioned_tables, sql_identifier_tokens


@dataclass(frozen=True, slots=True)
class _Table:
    columns: list[tuple[str, str]]  # (name, declared type)
    primary_key: frozenset[str]
    foreign_keys: list[tuple[str, str, str]]  # (column, parent table, parent column)
    create_sql: str


@lru_cache(maxsize=32768)  # SynSQL has 16.5K schemas; 2048 thrashed on the full set
def _graph(db_path: str) -> dict[str, _Table]:
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        tables: dict[str, _Table] = {}
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name"
        ).fetchall()
        for name, create_sql in rows:
            info = con.execute(f'PRAGMA table_info("{name}")').fetchall()
            fks = [
                (row[3], row[2], row[4])
                for row in con.execute(f'PRAGMA foreign_key_list("{name}")')
            ]
            tables[name] = _Table(
                columns=[(row[1], row[2] or "") for row in info],
                primary_key=frozenset(row[1] for row in info if row[5]),
                foreign_keys=fks,
                create_sql=create_sql,
            )
        return tables
    finally:
        con.close()


def _keep_tables(sql: str, graph: dict[str, _Table]) -> list[str]:
    """Query's tables plus their degree-1 FK parents and children."""
    schema = {name: [c for c, _ in t.columns] for name, t in graph.items()}
    mentioned = set(mentioned_tables(sql, schema))
    if not mentioned:
        return sorted(graph, key=str.lower)
    by_lower = {name.lower(): name for name in graph}
    keep = set(mentioned)
    for name in mentioned:
        for _, parent, _ in graph[name].foreign_keys:  # parents of mentioned
            resolved = by_lower.get(parent.lower())
            if resolved:
                keep.add(resolved)
    for name, table in graph.items():  # children pointing at mentioned
        for _, parent, _ in table.foreign_keys:
            if by_lower.get(parent.lower()) in mentioned:
                keep.add(name)
    return sorted(keep, key=str.lower)


def _rebuild(name: str, table: _Table, kept: list[str], keep_set: set[str]) -> str:
    columns = ", ".join(f'"{col}" {typ}'.rstrip() for col, typ in table.columns if col in kept)
    clauses = [columns]
    if table.primary_key:
        pk = ", ".join(f'"{c}"' for c, _ in table.columns if c in table.primary_key)
        clauses.append(f"PRIMARY KEY ({pk})")
    for column, parent, parent_col in table.foreign_keys:
        if parent in keep_set or parent.lower() in {k.lower() for k in keep_set}:
            clauses.append(f'FOREIGN KEY ("{column}") REFERENCES "{parent}" ("{parent_col}")')
    return f'CREATE TABLE "{name}" ({", ".join(clauses)});'


def reduced_ddl(db_path: Path | str, sql: str, *, level: int) -> str:
    """Schema DDL reduced to *level* (0 verbatim / 1 tables / 2 columns)."""
    if level == 0:
        return schema_ddl(db_path)
    graph = _graph(str(db_path))
    keep = _keep_tables(sql, graph)
    if level == 1:
        return "\n\n".join(f"{graph[name].create_sql};" for name in keep)
    tokens = sql_identifier_tokens(sql)
    keep_set = set(keep)
    statements = []
    for name in keep:
        table = graph[name]
        fk_cols = {column for column, _, _ in table.foreign_keys}
        kept = [
            col
            for col, _ in table.columns
            if col.lower() in tokens or col in table.primary_key or col in fk_cols
        ]
        statements.append(_rebuild(name, table, kept, keep_set))
    return "\n\n".join(statements)
