"""Turn examples + a generator into a score-ready predictions file, with receipts.

The bridge between :mod:`sqlpup.eval.generate` and :mod:`sqlpup.eval.scorer`:
prompts are rendered from each example's *real* database schema through the
versioned :class:`~sqlpup.eval.prompts.PromptSpec`, and every run's meta
records the spec id, generator, and refine settings -- so any number a
predictions file later produces can be traced to exactly how it was made
(the measured-not-claimed receipt).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlpup.eval.compact import fit_ddl
from sqlpup.eval.dataset import BirdExample, resolve_db_path
from sqlpup.eval.execution import ExecutionScorer
from sqlpup.eval.generate import SQLGenerator, extract_sql
from sqlpup.eval.prompts import BIRD_DDL_V1, PromptSpec, schema_ddl
from sqlpup.eval.refine import refine_batch


def _git_sha() -> str | None:
    """The repo's HEAD sha when resolvable (provenance only, never required)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:  # pragma: no cover - git genuinely absent
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def generate_predictions(
    examples: Sequence[BirdExample],
    eval_dir: Path,
    generator: SQLGenerator,
    *,
    spec: PromptSpec = BIRD_DDL_V1,
    refine_retries: int | None = None,
    scorer: ExecutionScorer | None = None,
    block_constrain: bool = False,
    two_pass: bool = False,
    column_hints: bool = False,
    expand_fk: bool = False,
    self_consistency: int | None = None,
    temperature: float = 0.7,
    seed: int = 0,
    compact_overflow: bool = False,
    min_generation: int = 256,
    block_budget: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render, generate (optionally with execution-feedback refinement), record.

    Returns ``(records, meta)``: records are ``{"index", "predicted_sql"}``
    (plus ``"single_shot_sql"`` when refined -- both framings stay reportable),
    meta is the provenance block for the ``.meta.json`` sidecar.
    """
    if refine_retries is not None and scorer is None:
        raise ValueError("refine_retries requires a scorer (the refine loop probes execution)")
    if refine_retries is not None and block_constrain:
        raise ValueError("block_constrain + refine is not wired yet; run them as separate passes")
    if two_pass and refine_retries is not None:
        raise ValueError("two_pass + refine is not wired yet; run them as separate passes")

    ddl_by_db: dict[str, str] = {}
    prompts: list[str] = []
    db_paths: list[Path] = []
    for example in examples:
        ddl = ddl_by_db.get(example.db_id)
        if ddl is None:
            ddl = schema_ddl(resolve_db_path(eval_dir, example.db_id))
            ddl_by_db[example.db_id] = ddl
        prompts.append(spec.render(ddl, question=example.question, evidence=example.evidence))
        db_paths.append(resolve_db_path(eval_dir, example.db_id))

    compaction_levels: dict[str, int] = {}
    vote_failures = 0
    if compact_overflow:
        compaction_levels = _compact_overflow(
            prompts, examples, db_paths, generator, spec=spec, min_generation=min_generation
        )

    records: list[dict[str, Any]]
    if self_consistency is not None:
        from sqlpup.eval.vote import majority_vote

        if scorer is None:
            raise ValueError("self_consistency requires a scorer (voting executes candidates)")
        sampler = getattr(generator, "generate_samples", None)
        if sampler is None:
            raise NotImplementedError(
                f"{type(generator).__name__} cannot sample; self-consistency needs k draws"
            )
        samples = sampler(prompts, self_consistency, temperature, seed)
        records = []
        for example, raws, db_path in zip(examples, samples, db_paths, strict=True):
            # Per-example isolation. A submission run is 1,534 questions in one
            # process; without this, one unfamiliar test-set schema that makes
            # voting raise discards every other answer too and the whole
            # submission reads as a failure. Falling back to the greedy
            # candidate keeps a real answer for that question and lets the run
            # finish; the count is reported so the failure is never silent.
            candidates: list[str] = []
            try:
                candidates = [extract_sql(spec.completion_prefix + raw) for raw in raws]
                winner, stats = majority_vote(candidates, db_path, scorer)
            except Exception as error:
                vote_failures += 1
                winner = candidates[0] if candidates else ""
                stats = {"error": type(error).__name__, "fell_back_to": "greedy"}
            records.append(
                {
                    "index": example.index,
                    "predicted_sql": winner,
                    "greedy_sql": candidates[0] if candidates else "",
                    "candidates": candidates,
                    "vote_stats": stats,
                }
            )
    elif two_pass:
        from sqlpup.eval.twopass import column_notes, focused_schema, tables_from_block

        first = generator.generate(prompts)
        narrowed: list[str] = []
        for example, completion, db_path in zip(examples, first, db_paths, strict=True):
            tables = tables_from_block(spec.completion_prefix + completion)
            ddl = focused_schema(db_path, tables, expand_fk=expand_fk)
            if column_hints:
                notes = column_notes(db_path, tables)
                if notes:
                    ddl = f"{ddl}\n\n{notes}"
            narrowed.append(spec.render(ddl, question=example.question, evidence=example.evidence))
        second = generator.generate(narrowed)
        records = [
            {
                "index": example.index,
                "predicted_sql": extract_sql(spec.completion_prefix + final),
                "raw_completion": spec.completion_prefix + final,
                # keep pass 1 so the two framings stay separately reportable
                "first_pass_sql": extract_sql(spec.completion_prefix + initial),
                "first_pass_tables": tables_from_block(spec.completion_prefix + initial),
            }
            for example, initial, final in zip(examples, first, second, strict=True)
        ]
    elif refine_retries is None:
        if block_budget is not None:
            budgeted = getattr(generator, "generate_block_budget", None)
            if budgeted is None:
                raise NotImplementedError(
                    f"{type(generator).__name__} cannot cap the linking block; "
                    "never silently decode without the lever that was asked for"
                )
            completions = budgeted(prompts, block_budget)
        elif block_constrain:
            constrained = getattr(generator, "generate_block_constrained", None)
            if constrained is None:
                raise NotImplementedError(
                    f"{type(generator).__name__} cannot decode block-constrained; "
                    "never silently decode unconstrained"
                )
            completions = constrained(prompts, db_paths)
        else:
            completions = generator.generate(prompts)
        records = [
            {
                "index": example.index,
                "predicted_sql": extract_sql(spec.completion_prefix + completion),
                # Kept because extraction discards the linking block, and the
                # block-vs-SQL divergence diagnostic needs to see it.
                "raw_completion": spec.completion_prefix + completion,
            }
            for example, completion in zip(examples, completions, strict=True)
        ]
    else:
        assert scorer is not None  # narrowed by the guard above
        results = refine_batch(
            prompts, db_paths, generator, scorer, spec=spec, max_retries=refine_retries
        )
        records = [
            {
                "index": example.index,
                "predicted_sql": result.final_sql,
                "single_shot_sql": result.single_shot_sql,
            }
            for example, result in zip(examples, results, strict=True)
        ]

    meta: dict[str, Any] = {
        "prompt_spec": spec.spec_id,
        "generator": type(generator).__name__,
        "refine_retries": refine_retries,
        "block_constrained": block_constrain,
        "two_pass": two_pass,
        "column_hints": column_hints,
        "expand_fk": expand_fk,
        "self_consistency": self_consistency,
        "examples": len(records),
        "compact_overflow": compact_overflow,
        "block_budget": block_budget,
        "vote_failures": vote_failures,
        "compacted": sum(compaction_levels.values()),
        "compaction_levels": compaction_levels,
        "git_sha": _git_sha(),
    }
    return records, meta


def _compact_overflow(
    prompts: list[str],
    examples: Sequence[BirdExample],
    db_paths: Sequence[Path],
    generator: SQLGenerator,
    *,
    spec: PromptSpec,
    min_generation: int,
) -> dict[str, int]:
    """Re-render, in place, the prompts that would otherwise be dropped.

    Only prompts at or over the generator's context are touched, so a run's
    results are bit-identical to a plain run except where the alternative was
    a guaranteed empty prediction. Returns a level histogram for the meta
    sidecar, because a prompt change we did not disclose is exactly what
    invalidates a leaderboard entry.
    """
    count_tokens = getattr(generator, "count_tokens", None)
    context_limit = getattr(generator, "context_limit", None)
    if not callable(count_tokens) or not isinstance(context_limit, int):
        raise ValueError(
            "compact_overflow needs a generator exposing a token budget "
            "(count_tokens() and context_limit); "
            f"{type(generator).__name__} exposes neither"
        )

    levels: dict[str, int] = {}
    for position, (example, db_path) in enumerate(zip(examples, db_paths, strict=True)):
        # Not just hard overflow: a prompt that fits but leaves only a handful of
        # positions to answer in produces a truncated linking block and no SQL,
        # which scores identically to an empty prediction. Measured on full dev,
        # that near-miss band is 36 of 117 empties.
        if context_limit - count_tokens(prompts[position]) >= min_generation:
            continue
        # Budget the schema alone: what the rest of the rendered prompt leaves.
        scaffold = spec.render("", question=example.question, evidence=example.evidence)
        room = context_limit - count_tokens(scaffold) - min_generation
        ddl, level = fit_ddl(db_path, budget=max(room, 0), measure=count_tokens)
        prompts[position] = spec.render(ddl, question=example.question, evidence=example.evidence)
        levels[str(level)] = levels.get(str(level), 0) + 1
    return levels
