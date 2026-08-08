"""Deterministic column requalification -- the post-fix for the 77%% bucket.

v2e's error taxonomy: 77%% of execution errors are ``no such column: T.c``
where column ``c`` exists but belongs to a *different* table in the same
query (measured example: ``T1.CustomerID FROM gasstations`` -- CustomerID
lives in the joined ``yearmonth``). The repair is mechanical: when exactly
one table in the query owns the column, requalify the reference. It never
guesses: ambiguous or truly-absent columns return None and the error stands.
"""

from __future__ import annotations

from sqlpup.eval.repair import requalify_columns

SCHEMA = {
    "gasstations": ["GasStationID", "ChainID", "Country"],
    "yearmonth": ["CustomerID", "Date", "Consumption"],
    "transactions_1k": ["TransactionID", "CustomerID", "Date"],
}


def test_requalifies_column_to_the_joined_table_that_owns_it() -> None:
    sql = (
        "SELECT T1.CustomerID FROM gasstations AS T1 "
        "INNER JOIN yearmonth AS T2 ON T1.GasStationID = T2.Date"
    )
    fixed = requalify_columns(sql, "no such column: T1.CustomerID", SCHEMA)
    assert fixed == (
        "SELECT T2.CustomerID FROM gasstations AS T1 "
        "INNER JOIN yearmonth AS T2 ON T1.GasStationID = T2.Date"
    )


def test_ambiguous_column_owned_by_two_joined_tables_is_not_guessed() -> None:
    sql = (
        "SELECT T1.CustomerID FROM gasstations AS T1 "
        "INNER JOIN yearmonth AS T2 ON 1=1 INNER JOIN transactions_1k AS T3 ON 1=1"
    )
    assert requalify_columns(sql, "no such column: T1.CustomerID", SCHEMA) is None


def test_column_absent_from_every_joined_table_returns_none() -> None:
    sql = "SELECT T1.Nonexistent FROM gasstations AS T1"
    assert requalify_columns(sql, "no such column: T1.Nonexistent", SCHEMA) is None


def test_non_column_errors_return_none() -> None:
    assert requalify_columns("SELECT 1", 'near ")": syntax error', SCHEMA) is None


def test_all_occurrences_of_the_bad_reference_are_rewritten() -> None:
    sql = (
        "SELECT T1.CustomerID FROM gasstations AS T1 "
        "INNER JOIN yearmonth AS T2 ON 1=1 "
        "WHERE T1.CustomerID > 5 ORDER BY T1.CustomerID"
    )
    fixed = requalify_columns(sql, "no such column: T1.CustomerID", SCHEMA)
    assert fixed is not None
    assert "T1.CustomerID" not in fixed
    assert fixed.count("T2.CustomerID") == 3
