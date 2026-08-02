"""Linking-block targets: name the schema objects before writing the SQL.

v2e's probe measured linking F1 0.978 on correct answers vs 0.667 on wrong
ones -- the model that says its tables/columns out loud first is the model
that binds them correctly. ``linked_target`` derives that block from the gold
SQL against the real schema, so training targets teach explicit binding and
the block is verifiable (only identifiers that truly exist may appear).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlpup.sft.linking import linked_target, schema_identifiers


def _toy_db(tmp_path: Path) -> Path:
    db = tmp_path / "toy" / "toy.sqlite"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL);
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        """
    )
    con.commit()
    con.close()
    return db


def test_schema_identifiers_maps_tables_to_columns(tmp_path: Path) -> None:
    schema = schema_identifiers(_toy_db(tmp_path))
    assert schema == {
        "orders": ["id", "customer_id", "total"],
        "customers": ["id", "name", "city"],
    }


def test_linked_target_names_only_identifiers_present_in_sql(tmp_path: Path) -> None:
    schema = schema_identifiers(_toy_db(tmp_path))
    sql = "SELECT T2.name FROM orders AS T1 INNER JOIN customers AS T2 ON T1.customer_id = T2.id"
    block = linked_target(sql, schema)
    assert block == (
        "-- tables: customers, orders\n"
        "-- columns: customers.id, customers.name, orders.customer_id\n"
    )


def test_linked_target_is_case_insensitive_but_emits_schema_case(tmp_path: Path) -> None:
    schema = schema_identifiers(_toy_db(tmp_path))
    block = linked_target("select NAME from CUSTOMERS", schema)
    assert block == "-- tables: customers\n-- columns: customers.name\n"


def test_linked_target_ignores_columns_of_unmentioned_tables(tmp_path: Path) -> None:
    schema = schema_identifiers(_toy_db(tmp_path))
    # "id" exists in both tables; only the mentioned table's column is claimed.
    block = linked_target("SELECT id FROM orders", schema)
    assert block == "-- tables: orders\n-- columns: orders.id\n"


def test_linked_target_empty_when_nothing_binds(tmp_path: Path) -> None:
    schema = schema_identifiers(_toy_db(tmp_path))
    assert linked_target("SELECT 1", schema) == "-- tables:\n-- columns:\n"
