"""The single-example execution-match core -- the unit the GRPO reward reuses.

:class:`ExecutionScorer` owns one killable SQLite worker (see
:mod:`sqlpup.eval.sandbox`) and scores ``(predicted, gold, db)`` triples
sequentially, reusing the worker across calls so the RL reward pays the process
spawn cost once, not per query. :func:`execution_match` is the convenience
wrapper for one-off use and for expressing the pure reward contract
``(pred, gold, db) -> MatchResult``.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Final

from sqlpup.eval.results import MatchResult
from sqlpup.eval.sandbox import SqliteWorker

# Per-query wall-clock budget (design spec section 6: 5s).
DEFAULT_TIMEOUT: Final = 5.0
# Distinct-row memory cap. The largest gold result set in BIRD dev is 228,765
# distinct rows (card_games, question_id 384), so this default sits well above
# every dev/mini-dev gold set -- preserving official set-equality for all of
# them -- while still hard-bounding a runaway prediction's memory in the worker.
DEFAULT_ROW_LIMIT: Final = 1_000_000


class ExecutionScorer:
    """Scores predictions against gold with one reused killable worker."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        row_limit: int = DEFAULT_ROW_LIMIT,
    ) -> None:
        self._timeout = timeout
        self._row_limit = row_limit
        self._worker = SqliteWorker()

    def score(self, predicted_sql: str, gold_sql: str, db_path: Path | str) -> MatchResult:
        """Execute both queries read-only and compare result sets (unordered)."""
        match, category = self._worker.evaluate(
            str(db_path), gold_sql, predicted_sql, self._timeout, self._row_limit
        )
        return MatchResult(match=match, category=category)

    def probe(self, sql: str, db_path: Path | str) -> tuple[bool, str]:
        """Execute ``sql`` alone (no gold): ``(ok, error_message)``.

        The refine loop's signal: the error text (missing column, syntax
        error, timeout) is what gets shown back to the model in a repair
        prompt. Uses the same killable worker, timeout and row cap as
        :meth:`score`.
        """
        return self._worker.probe(str(db_path), sql, self._timeout, self._row_limit)

    def fingerprint(self, sql: str, db_path: Path | str) -> tuple[bool, str]:
        """``(ok, digest)`` of ``sql``'s result set under the same sandbox.

        Self-consistency compares *answers*, not query text, so voting needs
        a stable identity for "the rows this query returns".
        """
        return self._worker.fingerprint(str(db_path), sql, self._timeout, self._row_limit)

    def close(self) -> None:
        self._worker.close()

    def __enter__(self) -> ExecutionScorer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def execution_match(
    predicted_sql: str,
    gold_sql: str,
    db_path: Path | str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> MatchResult:
    """Score a single ``(predicted, gold, db)`` triple (spins up a short-lived worker).

    This is the reward-reusable contract. For scoring many triples, hold an
    :class:`ExecutionScorer` and call :meth:`ExecutionScorer.score` so the worker
    is spawned once.
    """
    with ExecutionScorer(timeout=timeout, row_limit=row_limit) as scorer:
        return scorer.score(predicted_sql, gold_sql, db_path)
