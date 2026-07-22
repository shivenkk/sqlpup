"""Typed outcome of a single execution-match comparison.

The verdict (``match``) is exactly the official BIRD rule -- ``set(predicted
rows) == set(gold rows)`` -- while ``category`` refines *why* into a tag that is
diagnostic only: every category still reduces to the same 0/1 the official
evaluator would produce. ``MatchCategory`` is a :class:`enum.StrEnum` so its
members serialise straight to their string value in JSON reports.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class MatchCategory(enum.StrEnum):
    """Why a comparison landed where it did (diagnostic; never flips the verdict).

    ``OK``
        Both queries executed and their result sets were compared.
    ``EMPTY``
        Both result sets were empty -> a match under official semantics.
    ``EXECUTION_ERROR``
        The *predicted* query raised (syntax error, unknown table, a rejected
        write/ATTACH, ...) -> non-match.
    ``TIMEOUT``
        The *predicted* query exceeded the per-query wall-clock budget and its
        worker was killed -> non-match.
    ``ROW_LIMIT``
        The *predicted* query's distinct rows exceeded the memory cap while gold
        stayed under it, so the sets cannot be equal -> non-match.
    ``GOLD_ERROR``
        The *gold* query failed (raised, timed out, or blew the cap). A harness
        problem, surfaced loudly so it is never silently scored as a wrong
        answer -> non-match.
    """

    OK = "ok"
    EMPTY = "empty"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    ROW_LIMIT = "row_limit"
    GOLD_ERROR = "gold_error"


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The verdict for one ``(predicted, gold, db)`` comparison."""

    match: bool
    category: MatchCategory
