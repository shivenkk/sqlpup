"""Turn BIRD examples into GRPO rollout rows.

TRL wants a dataset with a ``prompt`` column; every other column is forwarded
verbatim to the reward function, so ``gold_sql`` and ``db_path`` ride along and
the reward stays a pure function of the row.

Two invariants are enforced here because nothing downstream would catch them:

**The prompt is rendered by the same PromptSpec evaluation uses.** Training on
one format and scoring under another is invisible to the loss.

**Every prompt leaves ``max_completion_length`` positions free.** TRL 1.9
removed ``max_prompt_length``, so a prompt filling the context produces
truncated rollouts that score 0 no matter what the model knows -- pure gradient
noise. Prompts that need room are compacted with the same gold-blind ladder
measured on the development set; prompts that cannot fit even then are dropped
and counted rather than truncated, since a half-schema prompt teaches the model
to answer from a schema it will never be shown.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sqlpup.eval.compact import fit_ddl
from sqlpup.eval.dataset import BirdExample
from sqlpup.eval.prompts import BIRD_DDL_V1, PromptSpec, schema_ddl


def build_rollout_rows(
    examples: Sequence[BirdExample],
    db_root: Path | str,
    *,
    count_tokens: Callable[[str], int],
    context_limit: int,
    max_completion_length: int,
    spec: PromptSpec = BIRD_DDL_V1,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """``(rows, stats)`` -- rows are TRL-ready, stats are for the run receipt.

    ``db_root`` follows BIRD's layout, ``<root>/<db_id>/<db_id>.sqlite``, which
    is shared by ``train_databases`` and ``dev_databases``.
    """
    if max_completion_length < 1:
        raise ValueError(f"max_completion_length must be >= 1, got {max_completion_length}")
    if context_limit <= max_completion_length:
        raise ValueError(
            f"context_limit ({context_limit}) must exceed "
            f"max_completion_length ({max_completion_length}) or no prompt can fit"
        )

    budget = context_limit - max_completion_length
    root = Path(db_root)
    rows: list[dict[str, str]] = []
    stats = {"kept": 0, "compacted": 0, "dropped": 0, "missing_db": 0}

    for example in examples:
        db_path = root / example.db_id / f"{example.db_id}.sqlite"
        if not db_path.exists():
            stats["missing_db"] += 1
            continue

        prompt = spec.render(
            schema_ddl(db_path), question=example.question, evidence=example.evidence
        )
        if count_tokens(prompt) > budget:
            scaffold = count_tokens(
                spec.render("", question=example.question, evidence=example.evidence)
            )
            ddl, _level = fit_ddl(db_path, budget=max(budget - scaffold, 0), measure=count_tokens)
            prompt = spec.render(ddl, question=example.question, evidence=example.evidence)
            if count_tokens(prompt) > budget:
                stats["dropped"] += 1
                continue
            stats["compacted"] += 1

        stats["kept"] += 1
        rows.append(
            {
                "prompt": prompt,
                "gold_sql": example.gold_sql,
                "db_path": str(db_path),
            }
        )
    return rows, stats


def to_hf_dataset(rows: Sequence[dict[str, str]]) -> Any:
    """Wrap rows as a ``datasets.Dataset`` (imported lazily; TRL-only path)."""
    from datasets import Dataset

    return Dataset.from_list(list(rows))
