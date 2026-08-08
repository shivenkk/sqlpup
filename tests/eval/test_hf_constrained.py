"""Batched constraint enforcement inside HF generation.

The processor applies each row's ``SQLConstraint`` mask to that row's logits:
disallowed vocabulary goes to ``-inf`` so greedy decoding cannot pick it. Rows
are independent -- one example's schema never leaks into a batch-mate's mask.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sqlpup.eval.hf_generator import _ConstraintProcessor  # noqa: E402


class _OnlyToken:
    """Scripted constraint: exactly one vocabulary id is ever legal."""

    def __init__(self, allowed_id: int) -> None:
        self.allowed_id = allowed_id
        self.seen_prefixes: list[list[int]] = []

    def allowed_token_mask(self, prefix_token_ids, vocab_size):  # type: ignore[no-untyped-def]
        self.seen_prefixes.append(list(prefix_token_ids))
        mask = [False] * vocab_size
        mask[self.allowed_id] = True
        return mask


def test_each_row_masked_by_its_own_constraint() -> None:
    c0, c1 = _OnlyToken(2), _OnlyToken(5)
    processor = _ConstraintProcessor([c0, c1], padded_prompt_len=3)
    input_ids = torch.tensor([[9, 9, 9, 7], [9, 9, 9, 8]])  # 3 prompt + 1 generated
    scores = torch.zeros((2, 6))
    out = processor(input_ids, scores)
    assert out[0].argmax().item() == 2
    assert out[1].argmax().item() == 5
    assert torch.isinf(out[0][3]) and out[0][3] < 0
    # each constraint saw only its own row's GENERATED suffix (prompt sliced off)
    assert c0.seen_prefixes == [[7]]
    assert c1.seen_prefixes == [[8]]


def test_all_true_mask_leaves_scores_untouched() -> None:
    class _Free:
        def allowed_token_mask(self, prefix_token_ids, vocab_size):  # type: ignore[no-untyped-def]
            return [True] * vocab_size

    processor = _ConstraintProcessor([_Free()], padded_prompt_len=0)
    scores = torch.full((1, 4), 1.5)
    out = processor(torch.tensor([[1]]), scores)
    assert torch.equal(out, torch.full((1, 4), 1.5))


def test_a_constraint_that_admits_nothing_must_not_wedge_the_decoder() -> None:
    """Measured failure (v3 step-30K): an all-False mask sends every logit to
    -inf, so greedy decoding emits garbage for the rest of the sequence: EX
    collapsed to 4.4% and validity *fell*, which a constraint can never
    legitimately cause. No admissible token means decode unconstrained."""

    class _AdmitsNothing:
        def allowed_token_mask(self, prefix_token_ids, vocab_size):  # type: ignore[no-untyped-def]
            return [False] * vocab_size

    processor = _ConstraintProcessor([_AdmitsNothing()], padded_prompt_len=0)
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = processor(torch.tensor([[7]]), scores)
    assert not torch.isinf(out).any()
    assert out.argmax().item() == 3  # the model's own preference survives


def test_consecutive_empty_completions_halt_the_run() -> None:
    """A silent failure (narrowing collapses to an empty schema, weights half
    loaded) produces empty completions forever. On a 1534-query submission run
    that wastes 10+ hours before a post-hoc check notices, so the guard lives
    inside the loop."""
    from sqlpup.eval.hf_generator import EmptyCompletionStreakError, guard_empty_streak

    streak = 0
    for _ in range(25):  # exactly at the limit, still tolerated
        streak = guard_empty_streak(streak, "", limit=25)
    assert streak == 25
    with pytest.raises(EmptyCompletionStreakError, match="silently"):
        guard_empty_streak(streak, "", limit=25)


def test_a_single_good_completion_resets_the_streak() -> None:
    from sqlpup.eval.hf_generator import guard_empty_streak

    streak = 0
    for _ in range(24):
        streak = guard_empty_streak(streak, "", limit=25)
    assert guard_empty_streak(streak, "SELECT 1", limit=25) == 0


def test_context_limit_can_be_overridden_below_the_model_config() -> None:
    """The Qwen control has a 32k context against our 2048. Left unchecked it
    would answer the long-schema questions we score as automatic misses,
    confounding 'what pretraining bought' with 'what a bigger window bought'.
    """
    from sqlpup.eval.hf_generator import _fit_new_tokens, _plan_batches

    # a prompt that fits 32k but not 2048 must be treated as overflow
    overflow, batches = _plan_batches([1500, 2500], batch_size=8, context_limit=2048)
    assert overflow == [1]
    assert batches == [[0]]
    assert _fit_new_tokens(512, prompt_length=1800, context_limit=2048) == 248


def test_generator_exposes_the_token_budget_the_compaction_fallback_needs() -> None:
    """``compact_overflow`` re-renders over-context prompts, so it must be able
    to ask the *real* tokenizer how long a candidate schema is. Duck-typed
    access to private attributes would break silently on refactor; these are
    the public seam, and predict.py raises if they are absent."""
    from sqlpup.eval.hf_generator import HFGreedyGenerator

    class _Tok:
        def __call__(self, text: str) -> dict[str, list[int]]:
            return {"input_ids": list(range(len(text) // 4))}

    generator = object.__new__(HFGreedyGenerator)
    generator._tokenizer = _Tok()  # type: ignore[assignment]
    generator._context_override = 2048

    assert generator.context_limit == 2048
    assert generator.count_tokens("x" * 40) == 10
