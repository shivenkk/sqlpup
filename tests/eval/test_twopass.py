"""Two-pass generation: let the model narrow its own schema, then answer.

Measured motivation (v3 step-15K): in 78% of binding failures the model's
linking block already names the right tables, and the query then fails on a
*column* inside them. Pass 1 harvests that block; pass 2 re-asks with only
those tables' DDL -- a handful of tables instead of forty -- optionally with
BIRD's column descriptions, which only fit once the schema is narrowed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlpup.eval.twopass import column_notes, focused_schema, tables_from_block


def _db(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    root.mkdir()
    con = sqlite3.connect(root / "shop.sqlite")
    con.executescript(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);"
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL);"
        "CREATE TABLE audit_log (id INTEGER PRIMARY KEY, entry TEXT);"
    )
    con.commit()
    con.close()
    return root / "shop.sqlite"


def test_tables_from_block_reads_the_tables_line() -> None:
    raw = "-- tables: orders, customers\n-- columns: orders.total\nSELECT 1"
    assert tables_from_block(raw) == ["orders", "customers"]


def test_tables_from_block_returns_empty_without_a_block() -> None:
    assert tables_from_block("SELECT 1 FROM t") == []
    assert tables_from_block("") == []


def test_focused_schema_keeps_only_named_tables(tmp_path: Path) -> None:
    db = _db(tmp_path)
    ddl = focused_schema(db, ["orders"])
    assert "orders" in ddl
    assert "audit_log" not in ddl
    assert "customers" not in ddl


def test_focused_schema_is_case_insensitive_and_ignores_unknown_names(tmp_path: Path) -> None:
    db = _db(tmp_path)
    ddl = focused_schema(db, ["ORDERS", "ghosts"])
    assert "orders" in ddl
    assert "ghosts" not in ddl


def test_focused_schema_falls_back_to_full_ddl_when_nothing_matches(tmp_path: Path) -> None:
    db = _db(tmp_path)
    ddl = focused_schema(db, ["ghosts"])
    # No usable narrowing -> the model must still see a real schema.
    assert "orders" in ddl and "customers" in ddl and "audit_log" in ddl


def test_column_notes_renders_bird_descriptions_for_named_tables(tmp_path: Path) -> None:
    db = _db(tmp_path)
    desc = db.parent / "database_description"
    desc.mkdir()
    (desc / "customers.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "city,City of residence,The city where the customer lives,text,\n"
        "id,,,integer,\n",
        encoding="utf-8-sig",
    )
    notes = column_notes(db, ["customers"])
    assert "customers.city" in notes
    assert "City of residence" in notes
    assert notes.startswith("--")  # comment lines, never executable text
    # a row with no human name and no description contributes nothing
    assert "customers.id" not in notes


def test_column_notes_is_empty_when_descriptions_are_absent(tmp_path: Path) -> None:
    assert column_notes(_db(tmp_path), ["customers"]) == ""


def test_column_notes_respects_a_line_budget(tmp_path: Path) -> None:
    db = _db(tmp_path)
    desc = db.parent / "database_description"
    desc.mkdir()
    rows = "\n".join(f"c{i},Human name {i},Description {i},text," for i in range(50))
    (desc / "customers.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        + rows,
        encoding="utf-8-sig",
    )
    notes = column_notes(db, ["customers"], max_lines=5)
    assert len(notes.splitlines()) == 5


def test_focused_schema_can_keep_fk_neighbours_of_the_named_tables(tmp_path: Path) -> None:
    """Gold-blind narrowing drops tables the query needed; FK expansion restores
    the join paths (measured: at step-30K the block named only *some* gold
    tables in ~37% of cases, and the moderate tier lost accuracy under strict
    narrowing while the simple tier gained)."""
    root = tmp_path / "shop2"
    root.mkdir()
    con = sqlite3.connect(root / "shop2.sqlite")
    con.executescript(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);"
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, "
        "customer_id INTEGER REFERENCES customers(id));"
        "CREATE TABLE audit_log (id INTEGER PRIMARY KEY, entry TEXT);"
    )
    con.commit()
    con.close()
    db = root / "shop2.sqlite"

    # match on the CREATE statement, not the bare name: orders' own DDL
    # mentions `REFERENCES customers(id)`, so a substring check always passes
    strict = focused_schema(db, ["orders"])
    assert "CREATE TABLE orders" in strict
    assert "CREATE TABLE customers" not in strict  # today's behaviour: only what was named

    expanded = focused_schema(db, ["orders"], expand_fk=True)
    assert "CREATE TABLE orders" in expanded
    assert "CREATE TABLE customers" in expanded  # FK parent restored
    assert "CREATE TABLE audit_log" not in expanded  # unrelated table still excluded


def test_column_notes_skips_lines_that_only_restate_the_column_name(tmp_path: Path) -> None:
    """Measured at step-30K: unfiltered notes were 71% of the schema block's
    characters and cost 4.2pt EX. Most lines were tautologies (`City: City`),
    so the fix is to emit only glosses that add information."""
    db = _db(tmp_path)
    desc = db.parent / "database_description"
    desc.mkdir()
    (desc / "customers.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "city,City,,text,\n"  # pure restatement
        "name,name,,text,\n"  # identical
        "id,Customer ID,,integer,\n"  # differs -> informative
        "code,CDS Code,The district-school identifier,text,\n",  # description -> informative
        encoding="utf-8-sig",
    )
    notes = column_notes(db, ["customers"])
    assert "customers.city" not in notes
    assert "customers.name" not in notes
    assert "customers.id" in notes
    assert "customers.code" in notes
