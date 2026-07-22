"""BIRD execution-accuracy evaluation harness (CPU-only, stdlib-only, torch-free).

The project's primary metric is BIRD execution accuracy (EX) under the official
evaluator's semantics: a prediction is correct iff its result set equals the
gold query's, compared as an unordered collection
(``set(predicted) == set(gold)``). The single-example core
(:func:`execution_match`) doubles as the GRPO reward, so nothing here imports
torch and everything runs on CPU with the standard library only.
"""

from __future__ import annotations

from sqlpup.eval.constrain import SQLConstraint
from sqlpup.eval.execution import (
    DEFAULT_ROW_LIMIT,
    DEFAULT_TIMEOUT,
    ExecutionScorer,
    execution_match,
)
from sqlpup.eval.generate import FakeGenerator, SQLGenerator, extract_sql
from sqlpup.eval.prompts import BIRD_DDL_V1, PromptSpec, schema_ddl
from sqlpup.eval.refine import DEFAULT_MAX_RETRIES, Attempt, RefineResult, refine_batch
from sqlpup.eval.results import MatchCategory, MatchResult

__all__ = [
    "BIRD_DDL_V1",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_ROW_LIMIT",
    "DEFAULT_TIMEOUT",
    "Attempt",
    "ExecutionScorer",
    "FakeGenerator",
    "MatchCategory",
    "MatchResult",
    "PromptSpec",
    "RefineResult",
    "SQLConstraint",
    "SQLGenerator",
    "execution_match",
    "extract_sql",
    "refine_batch",
    "schema_ddl",
]
