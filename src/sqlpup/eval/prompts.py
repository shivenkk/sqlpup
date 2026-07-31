"""Prompt rendering for BIRD evaluation -- the SFT/eval shared contract.

The rendered string is the *format contract* between fine-tuning and
evaluation: SFT examples and eval prompts must be byte-identical in shape, so
the format lives in one place, is versioned by name, and is content-addressed
by :attr:`PromptSpec.spec_id` (recorded in eval reports; a silent format drift
changes the id). Schema DDL is taken verbatim from ``sqlite_master`` -- the
same surface the model saw in pretraining (SchemaPile / StarCoder SQL) --
ordered by table name so rendering is deterministic across platforms.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Verbatim per-table DDL; name-ordered for determinism; internal sqlite_*
# bookkeeping tables carry no schema signal and are excluded.
_DDL_QUERY: Final = (
    "SELECT sql FROM sqlite_master "
    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
    "ORDER BY name"
)


def schema_ddl(db_path: Path | str) -> str:
    """The database's ``CREATE TABLE`` statements, verbatim and name-ordered."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(_DDL_QUERY).fetchall()
    finally:
        con.close()
    return "\n\n".join(f"{sql};" for (sql,) in rows)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """A named, content-addressed prompt format (see module docstring)."""

    name: str
    template: str  # fields: {ddl} {evidence_block} {question}
    evidence_line: str  # field: {evidence}; omitted entirely when evidence is blank
    repair_template: str  # fields: {failed_sql} {error}; appended by render_repair
    # Text the template's cue already began (e.g. "SELECT") that must be glued
    # back onto the model's completion before extraction. A base model happily
    # *continues comment context* after a "-- SQL:" cue (measured: 0/1534 on
    # dev, every prediction itself a comment); ending the prompt mid-statement
    # forces statement mode. SFT-tuned checkpoints don't need the crutch, so
    # the plain variant keeps an empty prefix.
    completion_prefix: str = ""

    @property
    def spec_id(self) -> str:
        """``name@hash12`` -- changes iff any template content changes."""
        canon = "\x1f".join(
            (
                self.name,
                self.template,
                self.evidence_line,
                self.repair_template,
                self.completion_prefix,
            )
        )
        digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]
        return f"{self.name}@{digest}"

    def render(self, ddl: str, question: str, evidence: str = "") -> str:
        """Render one example's prompt; blank evidence drops the whole line."""
        evidence_block = self.evidence_line.format(evidence=evidence) if evidence.strip() else ""
        return self.template.format(ddl=ddl, evidence_block=evidence_block, question=question)

    def render_repair(self, prompt: str, failed_sql: str, error: str) -> str:
        """Extend a prompt with a failed attempt + error, ending in a fresh cue.

        Rounds accumulate: passing a round-1 repair prompt back in yields a
        round-2 prompt that shows the model its whole failure history.
        """
        return prompt + self.repair_template.format(failed_sql=failed_sql, error=error)


BIRD_DDL_V1: Final = PromptSpec(
    name="bird-ddl-v1",
    template="-- SQLite schema:\n{ddl}\n\n{evidence_block}-- Question: {question}\n-- SQL:\n",
    evidence_line="-- External knowledge: {evidence}\n",
    repair_template="{failed_sql}\n-- The query above failed: {error}\n-- Corrected SQL:\n",
)

# Base-model variant: the cue ends mid-statement (see completion_prefix note).
BIRD_DDL_V1_SELECTCUE: Final = PromptSpec(
    name="bird-ddl-v1-selectcue",
    template=(
        "-- SQLite schema:\n{ddl}\n\n{evidence_block}-- Question: {question}\n-- SQL:\nSELECT"
    ),
    evidence_line="-- External knowledge: {evidence}\n",
    repair_template="{failed_sql}\n-- The query above failed: {error}\n-- Corrected SQL:\nSELECT",
    completion_prefix="SELECT",
)

# Name -> spec, for CLI selection and report provenance round-trips.
SPECS: Final[dict[str, PromptSpec]] = {
    spec.name: spec for spec in (BIRD_DDL_V1, BIRD_DDL_V1_SELECTCUE)
}
