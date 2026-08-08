"""Execution-feedback self-correction: probe errors become repair prompts.

The loop only ever sees *execution* feedback (the sandbox probe's error text)
-- never the gold answer -- so refined numbers remain honest. Every attempt is
retained per example so reports can state single-shot EX and refined EX side
by side; retries are capped and an identical regeneration short-circuits its
example (a model that repeats itself will keep repeating).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sqlpup.eval.execution import ExecutionScorer
from sqlpup.eval.generate import SQLGenerator, extract_sql
from sqlpup.eval.prompts import BIRD_DDL_V1, PromptSpec

# Addendum-locked default: report single-shot and <=2-retry refined EX.
DEFAULT_MAX_RETRIES: Final = 2


@dataclass(frozen=True, slots=True)
class Attempt:
    """One generated SQL and what happened when it was executed alone."""

    sql: str
    executed_ok: bool
    error: str  # "" when executed_ok


@dataclass(frozen=True, slots=True)
class RefineResult:
    """Every attempt for one example, first to last."""

    index: int
    attempts: tuple[Attempt, ...]

    @property
    def single_shot_sql(self) -> str:
        return self.attempts[0].sql

    @property
    def final_sql(self) -> str:
        return self.attempts[-1].sql

    @property
    def used_retries(self) -> int:
        return len(self.attempts) - 1


def refine_batch(
    prompts: Sequence[str],
    db_paths: Sequence[Path | str],
    generator: SQLGenerator,
    scorer: ExecutionScorer,
    *,
    spec: PromptSpec = BIRD_DDL_V1,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[RefineResult]:
    """Generate for every prompt, then re-prompt only still-failing examples.

    Round 0 generates for all examples; each subsequent round extends the
    failing examples' prompts with their last attempt + error
    (:meth:`PromptSpec.render_repair`) and regenerates for just those. An
    example leaves the loop when its SQL executes cleanly, when a retry
    regenerates the identical SQL, or when ``max_retries`` rounds are spent.
    """
    if len(prompts) != len(db_paths):
        raise ValueError(f"{len(prompts)} prompts vs {len(db_paths)} db_paths")

    current_prompts = list(prompts)
    completions = generator.generate(current_prompts)
    attempts: list[list[Attempt]] = []
    for completion, db_path in zip(completions, db_paths, strict=True):
        sql = extract_sql(spec.completion_prefix + completion)
        ok, error = scorer.probe(sql, db_path)
        attempts.append([Attempt(sql=sql, executed_ok=ok, error=error)])

    active = [i for i in range(len(prompts)) if not attempts[i][-1].executed_ok]
    for _ in range(max_retries):
        if not active:
            break
        for i in active:
            last = attempts[i][-1]
            current_prompts[i] = spec.render_repair(current_prompts[i], last.sql, last.error)
        round_completions = generator.generate([current_prompts[i] for i in active])
        still_failing: list[int] = []
        for position, i in enumerate(active):
            sql = extract_sql(spec.completion_prefix + round_completions[position])
            if sql == attempts[i][-1].sql:  # repeating itself: stop this example
                continue
            ok, error = scorer.probe(sql, db_paths[i])
            attempts[i].append(Attempt(sql=sql, executed_ok=ok, error=error))
            if not ok:
                still_failing.append(i)
        active = still_failing

    return [
        RefineResult(index=i, attempts=tuple(example_attempts))
        for i, example_attempts in enumerate(attempts)
    ]
