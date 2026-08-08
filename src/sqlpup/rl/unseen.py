"""The examples supervised fine-tuning never saw.

Measured on the first GRPO pilot: training on BIRD-train opened at **0.769 mean
reward**, because v3's SFT oversampled that split ten times. The model was being
asked to improve on answers it had memorised, so 57% of rollout groups carried
zero advantage and the policy did not move (``kl`` ~3e-5). RL needs the
distribution where the model is *weak* -- which is the regime the 7.8pt headroom
was measured in on the development set.

SFT selects rows by a per-row seeded hash (``sha256(f"{seed}:{index}")``, kept
when the leading 8 bytes fall below the rate). That makes its complement
exactly computable rather than approximately: at ``--sample-rate 0.4 --seed 7``
it leaves 59.98% of SynSQL untouched, ~1.5M examples each with an executable
database.

The index must match the order ``prepare.py`` enumerated, which holds because
both stream the same file through :func:`iter_json_array`. An off-by-one here
would silently hand RL the memorised half again and reproduce the null result
at full cost, so the equivalence is asserted directly against ``_keep``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlpup.eval.dataset import BirdExample
from sqlpup.sft.prepare import _keep, iter_json_array


def iter_unseen_rows(
    data_path: Path | str,
    *,
    sample_rate: float,
    seed: int,
    limit: int | None = None,
) -> Iterator[BirdExample]:
    """Stream the examples an SFT run at ``(sample_rate, seed)`` did **not** take.

    ``sample_rate`` is the rate that SFT used, not the rate wanted here: this
    yields the complement. Streaming, because the corpus is multi-gigabyte and
    a pilot only needs a few hundred rows.
    """
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate}")

    taken = 0
    for index, row in enumerate(iter_json_array(Path(data_path))):
        if _keep(index, sample_rate, seed):
            continue  # SFT trained on this one
        sql = row.get("sql") or row.get("SQL") or row.get("query")
        if not sql:
            continue
        # SynSQL says external_knowledge where BIRD says evidence; dropping it
        # would train on prompts missing the hint evaluation prompts carry.
        evidence = row.get("external_knowledge") or row.get("evidence") or ""
        yield BirdExample(
            index=index,
            question_id=str(row.get("question_id", index)),
            db_id=str(row["db_id"]),
            question=str(row.get("question", "")),
            evidence=str(evidence),
            gold_sql=str(sql),
            difficulty=str(row.get("difficulty", "")),
        )
        taken += 1
        if limit is not None and taken >= limit:
            return
