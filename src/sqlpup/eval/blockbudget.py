"""Force the model out of its own linking block once it has spent enough on it.

Measured on full dev: 21 of the 117 empty predictions had ample context left but
emitted **only comment lines**. The model wrote ``-- tables: … / -- columns: …``
and never began a statement; 8 of the 21 ran to the 512-token cap in degenerate
repetition (``Enrollment (Enrollment (Enrollment (``). They concentrate in
schemas whose column names carry spaces and parentheses such as ``Enrollment (K-12)``
and ``Percent (%)``, which the block grammar has no way to quote, so the block
cannot be closed and the model never escapes it.

The block is scaffolding, not the answer: :func:`sqlpup.eval.generate.extract_sql`
discards it before scoring. A completion that is entirely block is therefore
worth exactly zero, however fluent it looks.

This constraint spends a fixed token budget on the block, then admits only a
newline, and immediately after that newline refuses ``--`` so a fresh comment
cannot restart the runaway. Once a non-comment line has begun it stands down
entirely: the SQL body is never constrained, which is the mistake that wedged
the decoder the first time constrained decoding was attempted here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

# Enough for a genuine two-line block on a wide schema, far short of the 512
# the runaway cases consumed.
DEFAULT_BLOCK_BUDGET: Final = 96


class BlockBudgetConstraint:
    """Caps how much of a completion may be spent inside the linking block."""

    def __init__(self, tokenizer: Any, *, prompt_len: int, budget_tokens: int) -> None:
        if budget_tokens < 1:
            raise ValueError(f"budget_tokens must be >= 1, got {budget_tokens}")
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len
        self._budget = budget_tokens
        self._newline_ids: list[int] | None = None
        self._comment_ids: list[int] | None = None

    def _vocab_texts(self) -> list[str]:
        getter = getattr(self._tokenizer, "token_texts", None)
        if callable(getter):
            texts: list[str] = list(getter())
            return texts
        from sqlpup.eval.blockmask import _token_texts

        return _token_texts(self._tokenizer)

    def _special_ids(self) -> tuple[list[int], list[int]]:
        if self._newline_ids is None or self._comment_ids is None:
            texts = self._vocab_texts()
            self._newline_ids = [i for i, t in enumerate(texts) if "\n" in t]
            # Any token that could open a comment line, so the escape cannot be
            # undone on the very next step.
            self._comment_ids = [i for i, t in enumerate(texts) if t.lstrip().startswith("--")]
        return self._newline_ids, self._comment_ids

    def allowed_token_mask(
        self, prefix_token_ids: Sequence[int], vocab_size: int
    ) -> Sequence[bool]:
        generated_ids = list(prefix_token_ids[self._prompt_len :])
        free = [True] * vocab_size
        if not generated_ids:
            return free

        text = self._tokenizer.decode(generated_ids)
        lines = text.split("\n")
        current = lines[-1]
        in_comment = current.lstrip().startswith("--")
        # A completion that already contains a non-comment, non-empty line has
        # started its statement; nothing here applies any more.
        started_sql = any(
            line.strip() and not line.lstrip().startswith("--") for line in lines[:-1]
        ) or (current.strip() and not in_comment)
        if started_sql:
            return free

        newline_ids, comment_ids = self._special_ids()

        if in_comment and len(generated_ids) > self._budget:
            mask = [False] * vocab_size
            for i in newline_ids:
                if i < vocab_size:
                    mask[i] = True
            # Never hand back an all-False mask: that sends every logit to -inf
            # and the decoder emits garbage for the rest of the sequence.
            return mask if any(mask) else free

        # Just escaped onto a fresh line while over budget: refuse to reopen.
        if not in_comment and len(generated_ids) > self._budget:
            mask = [True] * vocab_size
            for i in comment_ids:
                if i < vocab_size:
                    mask[i] = False
            return mask if any(mask) else free

        return free
