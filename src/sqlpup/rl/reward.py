"""The GRPO reward: executed match = 1.0, grounded-and-executes = 0.1, else 0.0.

Correctness dominates by an order of magnitude, but a small dense term keeps
early training from being all-zero reward -- a 400M model's first rollouts
rarely match gold, and group-relative advantages need within-group variance to
learn from. The bonus rides the sandbox's probe, so it carries the same
safety/timeout semantics as scoring (a write attempt or runaway query earns
nothing).

**The bonus requires touching the database.** Without that condition, ``SELECT
1`` earns it (measured: so do ``SELECT 'x'`` and ``SELECT 1 AS a``), which is a
degenerate attractor under group-relative advantage -- a constant string that
always beats broken SQL, costs almost no tokens, and requires no schema
understanding. A small policy finds it quickly, and the collapse initially
*looks* like success: mean reward rises and completion length falls. Requiring
at least one real table keeps the dense signal pointed at the behaviour we
actually want, which is grounded SQL. The guard never touches a match: a gold
query that legitimately selects a constant still scores 1.0.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Protocol

from sqlpup.eval.execution import ExecutionScorer
from sqlpup.eval.generate import extract_sql
from sqlpup.sft.linking import mentioned_tables, schema_identifiers

logger = logging.getLogger(__name__)

# Deliberately an order of magnitude under a match: never worth trading
# correctness for executability, always worth emitting *grounded* SQL.
EXECUTES_BONUS: Final = 0.1


@lru_cache(maxsize=512)
def _schema(db_path: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Schema as a hashable snapshot -- rollouts hit the same databases nonstop."""
    return tuple((t, tuple(c)) for t, c in schema_identifiers(db_path).items())


def touches_database(sql: str, db_path: Path | str) -> bool:
    """Does *sql* reference at least one real table of this database?"""
    schema = {table: list(columns) for table, columns in _schema(str(db_path))}
    return bool(mentioned_tables(sql, schema))


def execution_reward(
    predicted_sql: str,
    gold_sql: str,
    db_path: Path | str,
    scorer: ExecutionScorer,
) -> float:
    """Score one rollout against gold on its database.

    The caller owns the :class:`ExecutionScorer` (one long-lived worker per
    rollout process, spawn cost paid once) -- the signature stays a pure
    ``(pred, gold, db) -> float`` contract for TRL's reward hook.
    """
    if not predicted_sql.strip():
        return 0.0
    if scorer.score(predicted_sql, gold_sql, db_path).match:
        return 1.0
    if not touches_database(predicted_sql, db_path):
        return 0.0
    executes_ok, _error = scorer.probe(predicted_sql, db_path)
    return EXECUTES_BONUS if executes_ok else 0.0


class RewardCallable(Protocol):
    """TRL's reward contract, plus the error counter a monitor reads."""

    __name__: str
    errors: int

    def __call__(self, **kwargs: Any) -> list[float | None]: ...


def make_execution_reward(scorer: ExecutionScorer) -> RewardCallable:
    """A TRL-shaped reward closing over a live :class:`ExecutionScorer`.

    TRL calls this as ``f(prompts=, completions=, completion_ids=, **columns)``
    where ``columns`` holds every non-reserved dataset column as a list aligned
    to the rollouts, alongside keys TRL injects itself (``trainer_state``,
    ``log_extra``, ``log_metric``). Hence ``**kwargs``: a stricter signature
    raises ``TypeError`` mid-step and kills the run.

    Completions are passed through :func:`extract_sql` first. v3 opens every
    completion with a ``-- tables:/-- columns:`` linking block, so scoring the
    raw text would score a comment and return 0 for every rollout -- a failure
    that looks exactly like "the model hasn't learned yet".
    """

    def execution_reward_fn(**kwargs: Any) -> list[float | None]:
        completions: Sequence[str] = kwargs["completions"]
        gold: Sequence[str] = kwargs["gold_sql"]
        databases: Sequence[str] = kwargs["db_path"]
        if not (len(completions) == len(gold) == len(databases)):
            raise ValueError(
                "reward columns must be aligned to the rollouts: "
                f"{len(completions)} completions, {len(gold)} gold_sql, "
                f"{len(databases)} db_path"
            )

        rewards: list[float | None] = []
        for completion, gold_sql, db_path in zip(completions, gold, databases, strict=True):
            try:
                sql = extract_sql(completion if isinstance(completion, str) else str(completion))
                rewards.append(execution_reward(sql, gold_sql, db_path, scorer))
            except Exception:  # one bad row must not end a 48h run
                # Counted rather than swallowed: the pilot alarms if the rate is
                # material, so this cannot quietly become "the reward is broken".
                execution_reward_fn.errors += 1  # type: ignore[attr-defined]
                logger.warning("reward failed for db=%s; scored 0.0", db_path, exc_info=True)
                rewards.append(0.0)
        return rewards

    execution_reward_fn.errors = 0  # type: ignore[attr-defined]
    execution_reward_fn.__name__ = "execution_reward"
    return execution_reward_fn  # type: ignore[return-value]


def make_exact_match_reward(scorer: ExecutionScorer) -> RewardCallable:
    """Rollout execution accuracy, for observation only.

    The training reward blends 1.0 matches with 0.1 grounding bonuses, so its
    mean cannot separate "the policy is sharpening" from "the policy is farming
    the bonus". That distinction matters most when ``beta`` is 0 and no KL
    anchor holds the policy near its starting point.

    Passed to TRL as a second reward function with weight 0: it is logged under
    its own name every step and contributes nothing to the gradient.
    """

    def rollout_ex_fn(**kwargs: Any) -> list[float | None]:
        completions: Sequence[str] = kwargs["completions"]
        gold: Sequence[str] = kwargs["gold_sql"]
        databases: Sequence[str] = kwargs["db_path"]
        out: list[float | None] = []
        for completion, gold_sql, db_path in zip(completions, gold, databases, strict=True):
            try:
                sql = extract_sql(completion if isinstance(completion, str) else str(completion))
                matched = bool(sql.strip()) and scorer.score(sql, gold_sql, db_path).match
                out.append(1.0 if matched else 0.0)
            except Exception:  # observation must never end a run
                out.append(0.0)
        return out

    rollout_ex_fn.errors = 0  # type: ignore[attr-defined]
    rollout_ex_fn.__name__ = "rollout_ex"
    return rollout_ex_fn  # type: ignore[return-value]
