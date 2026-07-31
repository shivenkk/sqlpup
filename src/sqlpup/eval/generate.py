"""The generator seam: prompts in, raw completions out, SQL extracted after.

``SQLGenerator`` is the one interface evaluation and the refine loop see; a
scripted :class:`FakeGenerator` drives all local tests, and the real
checkpoint-backed greedy generator is a thin adapter added when trained
weights exist (it lives behind the optional torch extra, never imported
here). Extraction is centralised in :func:`extract_sql` so every generator's
raw text is normalised identically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sqlpup.eval.constrain import SQLConstraint


class SQLGenerator(Protocol):
    """Batch text completion: one raw completion per prompt, in order."""

    def generate(self, prompts: Sequence[str]) -> list[str]: ...


def extract_sql(completion: str) -> str:
    """Normalise a raw completion to a single SQL statement.

    Unwraps one surrounding code fence if present, drops *leading* comment
    lines (v3 completions open with a ``-- tables:/-- columns:`` linking
    block that is scaffolding, not answer), cuts at the first ``;``
    (everything after it is model rambling), and strips whitespace. The
    trailing semicolon is dropped -- sqlite executes either form. Comments
    after the statement has begun are part of the SQL and stay.
    """
    text = completion.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline != -1 else ""
        closing = text.find("```")
        if closing != -1:
            text = text[:closing]
        text = text.strip()
    while text.startswith("--"):
        newline = text.find("\n")
        if newline == -1:
            return ""
        text = text[newline + 1 :].lstrip()
    semicolon = text.find(";")
    if semicolon != -1:
        text = text[:semicolon]
    return text.strip()


@dataclass
class FakeGenerator:
    """Scripted generator for tests: pops one response round per call.

    ``script`` is a list of rounds; round *k* must be exactly as wide as the
    *k*-th ``generate`` call's prompt batch. Calls are recorded for
    assertions about batching (e.g. the refine loop re-prompting only
    still-failing examples).
    """

    script: list[list[str]]
    constraint: SQLConstraint | None = None
    calls: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.constraint is not None:
            raise NotImplementedError(
                "SQL-constrained decoding is not implemented in this harness; "
                "it arrives with the batched inference engine (project #2)."
            )

    def generate(self, prompts: Sequence[str]) -> list[str]:
        self.calls.append(list(prompts))
        if not self.script:
            raise AssertionError("FakeGenerator script exhausted")
        round_outputs = self.script.pop(0)
        if len(round_outputs) != len(prompts):
            raise AssertionError(
                f"script round has {len(round_outputs)} outputs for {len(prompts)} prompts"
            )
        return round_outputs
