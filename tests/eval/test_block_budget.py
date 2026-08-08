"""Forcing the model out of its own linking block.

Measured on full dev: 21 of the 117 empty predictions had ample context but
emitted **only comment lines** -- the model wrote ``-- tables: ... / -- columns:
...`` and never began a statement, 8 of them running to the 512-token cap in
degenerate repetition (``Enrollment (Enrollment (Enrollment (``). They cluster
in schemas whose column names carry spaces and parentheses (``Enrollment
(K-12)``, ``Percent (%)``), which the block grammar cannot quote, so the block
never closes. Compaction incidentally rescued about eight of them; ~13 remain.

The block is scaffolding, not the answer: ``extract_sql`` discards it. So a
completion that is *entirely* block is worth exactly zero, and the fix is to
stop the model spending its budget there. Past a token budget this constraint
admits only a newline, and immediately after that newline it forbids starting
another comment -- so the next thing generated has to be SQL.

Decoding-level, deliberately: every prompt-reshaping lever tested on this model
lost, and both decoding-level levers won.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from sqlpup.eval.blockbudget import BlockBudgetConstraint


class _Vocab:
    """Minimal stand-in for a tokenizer: one id per surface string."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    def decode(self, ids):  # type: ignore[no-untyped-def]
        return "".join(self._tokens[i] for i in ids)

    def token_texts(self) -> list[str]:
        return list(self._tokens)


# ids:            0     1        2       3      4       5        6
TOKENS = ["-- tables: t", "\n", "--", " columns", "SELECT", " x", ","]
VOCAB = _Vocab(TOKENS)


def _mask(constraint: BlockBudgetConstraint, ids: Sequence[int]) -> Sequence[bool]:
    return constraint.allowed_token_mask(ids, len(TOKENS))


def test_under_budget_the_model_writes_its_block_freely() -> None:
    """The block is useful when it terminates: it is our schema-linking
    instrument and the two-pass input. This lever must not disturb it."""
    c = BlockBudgetConstraint(VOCAB, prompt_len=0, budget_tokens=8)
    assert all(_mask(c, [0]))


def test_past_the_budget_only_a_newline_is_admissible() -> None:
    c = BlockBudgetConstraint(VOCAB, prompt_len=0, budget_tokens=2)
    mask = _mask(c, [0, 2, 3])  # still inside a comment, over budget
    assert mask[1] is True  # newline
    assert not any(m for i, m in enumerate(mask) if i != 1)


def test_after_the_forced_newline_a_new_comment_is_refused() -> None:
    """Otherwise the model simply opens another ``--`` line and the runaway
    continues one line lower."""
    c = BlockBudgetConstraint(VOCAB, prompt_len=0, budget_tokens=2)
    mask = _mask(c, [0, 2, 3, 1])  # newline just emitted
    assert mask[2] is False  # "--" cannot start a fresh comment
    assert mask[4] is True  # SELECT is fine


def test_once_sql_has_started_the_constraint_stands_down() -> None:
    """The SQL body is never constrained -- that is what wedged the decoder the
    first time we tried constrained decoding."""
    c = BlockBudgetConstraint(VOCAB, prompt_len=0, budget_tokens=2)
    assert all(_mask(c, [0, 1, 4, 5]))


def test_a_completion_that_never_entered_a_block_is_untouched() -> None:
    """Not every completion opens with a block; those must decode normally."""
    c = BlockBudgetConstraint(VOCAB, prompt_len=0, budget_tokens=1)
    assert all(_mask(c, [4, 5, 5, 5]))


def test_the_prompt_is_not_counted_against_the_budget() -> None:
    """The budget is about the completion; counting the prompt would fire
    instantly on every schema-heavy prompt."""
    c = BlockBudgetConstraint(VOCAB, prompt_len=3, budget_tokens=4)
    assert all(_mask(c, [9, 9, 9, 0]))  # 3 prompt ids then one generated


@pytest.mark.parametrize("budget", [0, -1])
def test_a_nonsense_budget_is_rejected(budget: int) -> None:
    with pytest.raises(ValueError):
        BlockBudgetConstraint(VOCAB, prompt_len=0, budget_tokens=budget)


def test_it_never_returns_an_all_false_mask() -> None:
    """Measured failure (v3 step-30K): an all-False mask sends every logit to
    -inf and the decoder emits garbage for the rest of the sequence. Whatever
    the state, something must remain legal."""
    c = BlockBudgetConstraint(VOCAB, prompt_len=0, budget_tokens=1)
    for ids in ([0], [0, 2], [0, 2, 3], [0, 1], [0, 1, 4], [4], []):
        assert any(_mask(c, ids)), ids
