"""The checkpoint-backed generator adapter: model-free units only.

The real generation path is proven by the on-box integration smoke against
exported weights; here we pin the model-free contracts -- constraint rejection
happens before any (expensive, possibly-absent) model loading, and batching
math is exact.
"""

from __future__ import annotations

import pytest

from sqlpup.eval.hf_generator import HFGreedyGenerator, _plan_batches


def test_constraint_is_rejected_before_any_model_loading(tmp_path: object) -> None:
    class _NullConstraint:
        def allowed_token_mask(self, prefix_token_ids: object, vocab_size: int) -> list[bool]:
            return [True] * vocab_size

    # A nonexistent model dir would raise on loading; NotImplementedError proves
    # the constraint check fires first.
    with pytest.raises(NotImplementedError):
        HFGreedyGenerator("/nonexistent/model/dir", constraint=_NullConstraint())


def test_plan_batches_separates_overflow_and_sorts_by_length() -> None:
    lengths = [900, 2100, 1000, 2048, 100, 950]
    overflow, batches = _plan_batches(lengths, batch_size=2, context_limit=2048)
    assert overflow == [1, 3]  # at/over the context limit, in index order
    assert batches == [[4, 0], [5, 2]]  # ascending length, chunked
    covered = sorted(overflow + [i for batch in batches for i in batch])
    assert covered == list(range(len(lengths)))


def test_plan_batches_handles_empty_and_all_overflow() -> None:
    assert _plan_batches([], batch_size=4, context_limit=2048) == ([], [])
    overflow, batches = _plan_batches([3000, 2048], batch_size=4, context_limit=2048)
    assert overflow == [0, 1]
    assert batches == []


def test_plan_batches_rejects_a_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError):
        _plan_batches([100], batch_size=0, context_limit=2048)


def test_fit_new_tokens_respects_context_and_floor() -> None:
    from sqlpup.eval.hf_generator import _fit_new_tokens

    # Ample room: the requested budget stands.
    assert _fit_new_tokens(256, prompt_length=1000, context_limit=2048) == 256
    # Tight room: budget shrinks to what fits.
    assert _fit_new_tokens(256, prompt_length=1900, context_limit=2048) == 148
    # No room (prompt at/over the context): nothing can be generated.
    assert _fit_new_tokens(256, prompt_length=2048, context_limit=2048) == 0
    assert _fit_new_tokens(256, prompt_length=2139, context_limit=2048) == 0
