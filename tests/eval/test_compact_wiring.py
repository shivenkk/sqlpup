"""The compaction fallback, wired into prediction.

Default behaviour is unchanged: an over-context prompt is left alone and scores
as a miss, which is what every reported development number was measured under.
Opting in (``compact_overflow=True``, for submission runs) re-renders only the
prompts that would otherwise be dropped, using the gold-blind ladder.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sqlpup.eval.dataset import BirdExample
from sqlpup.eval.predict import generate_predictions


class _CountingGenerator:
    """Records what it was asked to generate; tokens are 1 per 4 characters."""

    context_limit = 100

    def __init__(self) -> None:
        self.seen: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def generate(self, prompts):  # type: ignore[no-untyped-def]
        self.seen = list(prompts)
        return ["SELECT 1" for _ in prompts]


def _eval_dir(tmp_path: Path, columns: int) -> Path:
    eval_dir = tmp_path / "eval"
    db_dir = eval_dir / "dev_databases" / "wide"
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / "wide.sqlite")
    body = ", ".join(f"column_with_a_long_name_{i} TEXT DEFAULT 'x'" for i in range(columns))
    con.execute(f"CREATE TABLE big (id INTEGER PRIMARY KEY, {body})")
    con.commit()
    con.close()
    (eval_dir / "dev.json").write_text(json.dumps([]))
    return eval_dir


def _example() -> BirdExample:
    return BirdExample(
        index=0,
        question_id="q0",
        db_id="wide",
        question="how many?",
        evidence="",
        gold_sql="SELECT 1",
        difficulty="simple",
    )


def test_overflow_prompt_is_dropped_by_default(tmp_path: Path) -> None:
    """The reported development numbers were measured this way; the fallback
    must not silently change what a plain run does."""
    eval_dir = _eval_dir(tmp_path, columns=200)
    generator = _CountingGenerator()
    _, meta = generate_predictions([_example()], eval_dir, generator)
    assert len(generator.seen) == 1  # rendered and handed over as-is
    assert meta.get("compacted") in (None, 0)


def test_compact_overflow_shrinks_the_prompt_until_it_fits(tmp_path: Path) -> None:
    eval_dir = _eval_dir(tmp_path, columns=200)
    generator = _CountingGenerator()
    plain = _CountingGenerator()
    generate_predictions([_example()], eval_dir, plain)
    generate_predictions([_example()], eval_dir, generator, compact_overflow=True)

    assert generator.count_tokens(generator.seen[0]) <= generator.context_limit
    assert len(generator.seen[0]) < len(plain.seen[0])
    assert "SELECT" in generator.seen[0] or "how many?" in generator.seen[0]


def test_a_prompt_that_already_fits_is_left_byte_identical(tmp_path: Path) -> None:
    """Compaction is a fallback, not a policy: touching prompts that fit would
    change results we have already measured and reported."""
    eval_dir = _eval_dir(tmp_path, columns=1)
    plain = _CountingGenerator()
    compacted = _CountingGenerator()
    generate_predictions([_example()], eval_dir, plain)
    generate_predictions(
        [_example()], eval_dir, compacted, compact_overflow=True, min_generation=16
    )
    assert plain.seen == compacted.seen


def test_meta_records_how_many_prompts_were_compacted(tmp_path: Path) -> None:
    """The submission write-up has to state this; an undisclosed prompt change
    is exactly the kind of thing that invalidates a leaderboard entry."""
    eval_dir = _eval_dir(tmp_path, columns=200)
    _, meta = generate_predictions(
        [_example()], eval_dir, _CountingGenerator(), compact_overflow=True
    )
    assert meta["compacted"] == 1
    assert sum(meta["compaction_levels"].values()) == 1


def test_generator_without_a_token_budget_is_rejected_loudly(tmp_path: Path) -> None:
    """Silently skipping the fallback would produce empty predictions on a
    submission run and we would find out from BIRD's abnormal-output check."""

    class _Bare:
        def generate(self, prompts):  # type: ignore[no-untyped-def]
            return ["SELECT 1" for _ in prompts]

    eval_dir = _eval_dir(tmp_path, columns=200)
    with pytest.raises(ValueError, match="token budget"):
        generate_predictions([_example()], eval_dir, _Bare(), compact_overflow=True)


def test_a_prompt_that_fits_but_cannot_be_answered_in_is_compacted(tmp_path: Path) -> None:
    """Measured on full dev: 36 of 117 empty predictions had prompts *under* the
    limit but only a handful of tokens left to answer in, so the model emitted a
    partial linking block and nothing else. Triggering only on hard overflow
    catches 60 of those 117; triggering on the generation budget catches 96."""
    eval_dir = _eval_dir(tmp_path, columns=200)

    class _Tight(_CountingGenerator):
        # prompt lands just under the limit -- the exact case that was missed
        context_limit = 620

    generator = _Tight()
    generate_predictions(
        [_example()], eval_dir, generator, compact_overflow=True, min_generation=256
    )
    used = generator.count_tokens(generator.seen[0])
    left = generator.context_limit - used
    assert left >= 256, f"only {left} tokens left to generate in"


# --- submission hardening: one bad example must not lose the whole run -------


class _SamplingGenerator(_CountingGenerator):
    """Generator that can sample k candidates, as self-consistency requires."""

    def generate_samples(self, prompts, k, temperature, seed):  # type: ignore[no-untyped-def]
        self.seen = list(prompts)
        return [["SELECT 1"] * k for _ in prompts]


class _ExplodingScorer:
    """Scores fine except on one database, where it raises."""

    def __init__(self, bad_db: str) -> None:
        self.bad_db = bad_db

    def score(self, predicted, gold, db_path):  # type: ignore[no-untyped-def]
        if str(db_path) == self.bad_db:
            raise RuntimeError("corrupt database page")
        raise AssertionError("unused in this test")

    def fingerprint(self, sql, db_path):  # type: ignore[no-untyped-def]
        if str(db_path) == self.bad_db:
            raise RuntimeError("corrupt database page")
        return "1:abc"


def test_one_unscorable_example_does_not_lose_the_whole_voting_run(tmp_path: Path) -> None:
    """A submission run is 1,534 questions in one process. If voting raises on a
    single unfamiliar test-set schema, an unguarded loop discards every other
    answer too and the submission reads as a total failure. The bad example
    falls back to its greedy candidate and the run continues."""
    from sqlpup.eval.dataset import resolve_db_path

    eval_dir = _eval_dir(tmp_path, columns=1)
    bad = str(resolve_db_path(eval_dir, "wide"))
    records, meta = generate_predictions(
        [_example()],
        eval_dir,
        _SamplingGenerator(),
        self_consistency=3,
        scorer=_ExplodingScorer(bad),  # type: ignore[arg-type]
    )
    assert len(records) == 1
    assert records[0]["predicted_sql"] == "SELECT 1"  # greedy fallback, not lost
    assert meta["vote_failures"] == 1
