"""Deterministic column requalification -- the disclosed post-fix layer.

The v2e taxonomy measured 77%% of execution errors as ``no such column: T.c``
where the column exists in a *different* table of the same query (the model
binds the right column to the wrong alias). When exactly one table mentioned
in the query owns the column, the repair is mechanical and provably safe:
requalify the reference and re-execute. Anything ambiguous or truly absent is
left alone -- this layer never guesses, so it can only convert
executes-with-error into executes; it cannot corrupt an already-valid query.
Applied at evaluation time and disclosed in the paper as a deterministic
post-fix (counted separately in ablations).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from sqlpup.sft.linking import _table_mentions  # alias map is the same problem

_NO_SUCH_COLUMN: Final = re.compile(r"no such column:\s*([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)")


def requalify_columns(sql: str, error: str, schema: Mapping[str, Sequence[str]]) -> str | None:
    """Rewrite ``bad_qualifier.column`` to the unique owning table's alias.

    Returns the repaired SQL, or None when the error is not a qualified
    no-such-column, the column's owner is ambiguous, or no mentioned table
    owns it.
    """
    match = _NO_SUCH_COLUMN.search(error)
    if match is None:
        return None
    bad_qualifier, column = match.group(1), match.group(2)
    mentioned, alias_to_table = _table_mentions(sql, schema)
    owners = [
        table for table in mentioned if any(c.lower() == column.lower() for c in schema[table])
    ]
    if len(owners) != 1:
        return None
    owner = owners[0]
    # Prefer the owner's alias if it has one distinct from the table name,
    # emitted in the casing the query itself uses.
    qualifier = next(
        (a for a, t in alias_to_table.items() if t == owner and a != owner.lower()),
        owner,
    )
    cased = re.search(rf"\b{re.escape(qualifier)}\b", sql, re.IGNORECASE)
    if cased is not None:
        qualifier = cased.group(0)
    pattern = re.compile(rf"\b{re.escape(bad_qualifier)}\s*\.\s*{re.escape(column)}\b")
    repaired = pattern.sub(f"{qualifier}.{column}", sql)
    return repaired if repaired != sql else None
