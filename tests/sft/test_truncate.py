"""Gold-aware DDL reduction: recover oversized pairs without severing JOINs.

22.5%% of SynSQL pairs and 10.4%% of BIRD pairs were *discarded* in v2e for
exceeding the 2048 context -- disproportionately the large-schema examples
where the errors live. Reduction trades schema completeness for fit in two
measured steps: level 1 drops tables unrelated to the query (keeping the
query's tables AND their degree-1 foreign-key neighbours, so JOIN paths the
model should learn stay visible); level 2 additionally strips unreferenced
non-key columns. Key columns (PK/FK) always survive.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlpup.sft.truncate import reduced_ddl


def _shop_db(tmp_path: Path) -> Path:
    db = tmp_path / "shop" / "shop.sqlite"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY, name TEXT, city TEXT, phone TEXT
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total REAL, note TEXT
        );
        CREATE TABLE audit_log (id INTEGER PRIMARY KEY, entry TEXT);
        """
    )
    con.commit()
    con.close()
    return db


def test_level0_returns_full_ddl_verbatim(tmp_path: Path) -> None:
    db = _shop_db(tmp_path)
    ddl = reduced_ddl(db, "SELECT total FROM orders", level=0)
    assert "audit_log" in ddl and "phone" in ddl and "note" in ddl


def test_level1_drops_unrelated_tables_but_keeps_fk_neighbours(tmp_path: Path) -> None:
    db = _shop_db(tmp_path)
    ddl = reduced_ddl(db, "SELECT total FROM orders", level=1)
    assert "orders" in ddl
    assert "customers" in ddl  # FK parent of orders: JOIN path preserved
    assert "audit_log" not in ddl  # unrelated: dropped
    assert "note" in ddl  # level 1 keeps all columns of surviving tables


def test_level2_strips_unreferenced_nonkey_columns(tmp_path: Path) -> None:
    db = _shop_db(tmp_path)
    ddl = reduced_ddl(db, "SELECT total FROM orders", level=2)
    assert "total" in ddl  # referenced by the query
    assert "customer_id" in ddl  # FK: always survives
    assert "note" not in ddl  # unreferenced, non-key: stripped
    assert "phone" not in ddl  # neighbour table also slimmed
    assert "customers" in ddl and '"id"' in ddl  # PKs survive


def test_level2_output_is_valid_executable_ddl(tmp_path: Path) -> None:
    db = _shop_db(tmp_path)
    ddl = reduced_ddl(db, "SELECT total FROM orders", level=2)
    con = sqlite3.connect(":memory:")
    con.executescript(ddl)  # must be well-formed CREATE TABLE statements
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert tables == {"customers", "orders"}


def test_reduction_is_deterministic(tmp_path: Path) -> None:
    db = _shop_db(tmp_path)
    assert reduced_ddl(db, "SELECT name FROM customers", level=2) == reduced_ddl(
        db, "SELECT name FROM customers", level=2
    )
