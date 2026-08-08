"""Block-constrained decoding: mask the linking block to real schema names.

The v3 target format opens every completion with::

    -- tables: customers, orders
    -- columns: customers.id, orders.customer_id

The block conditions the SQL that follows, so a hallucinated identifier there
cascades into the query (exposure bias). This constraint walks the decoded
block text and, at each step, permits only tokens that keep every named
identifier a prefix of (or exactly) a real schema name. After the block's
second newline the mask is all-True -- the SQL body is deliberately
unconstrained (that harder problem belongs to the full-grammar stage).

This is the reference implementation of the ``SQLConstraint`` protocol from
``eval/constrain.py``: correct and simple over fast. It re-decodes the
generated suffix each call and string-checks all vocabulary continuations
(~30ms/step at 32k vocab) -- fine for evaluation, and exactly the semantics
the batched engine (project #2) must reproduce quickly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_LINE_PREFIXES = ("-- tables:", "-- columns:")

# Vocab surface strings are identical for every constraint sharing a
# tokenizer; computing them is 32k id_to_token calls, so cache per tokenizer.
_TOKEN_TEXT_CACHE: dict[int, list[str]] = {}


def _split_list(text: str) -> tuple[list[str], str]:
    """Comma-separated identifiers so far, plus the one being typed.

    Splitting on ``", "`` cannot represent the moment after a comma is typed
    but before its space, which is a real decoding step -- that off-by-one
    forced every block to a single table (measured, v3 step-30K).
    """
    parts = [part.strip() for part in text.split(",")]
    return parts[:-1], parts[-1]


def _clean(token: str) -> str:
    """Convert a vocab surface form to its decoded text (BPE space markers)."""
    return token.replace("Ġ", " ").replace("Ċ", "\n")


def _token_texts(tokenizer: Any) -> list[str]:
    """Decoded per-id vocab strings for a ``tokenizers.Tokenizer`` OR an HF
    fast tokenizer (their lookup APIs differ; both are supported)."""
    cache_key = id(tokenizer)
    cached = _TOKEN_TEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if hasattr(tokenizer, "get_vocab_size"):  # tokenizers.Tokenizer
        vocab = tokenizer.get_vocab_size()
        lookup = tokenizer.id_to_token
        texts = [_clean(lookup(i) or "") for i in range(vocab)]
    else:  # transformers PreTrainedTokenizerFast
        vocab = len(tokenizer)
        tokens = tokenizer.convert_ids_to_tokens(list(range(vocab)))
        texts = [_clean(token or "") for token in tokens]
    _TOKEN_TEXT_CACHE[cache_key] = texts
    return texts


class LinkingBlockConstraint:
    """Token mask that confines the linking block to real schema identifiers."""

    def __init__(
        self,
        schema: Mapping[str, Sequence[str]],
        tokenizer: Any,  # tokenizers.Tokenizer or HF fast tokenizer (duck-typed decode)
        *,
        prompt_len: int,
    ) -> None:
        self._prompt_len = prompt_len
        self._tokenizer = tokenizer
        self._idents = (
            sorted(schema),  # line 0: table names
            sorted(f"{t}.{c}" for t, cols in schema.items() for c in cols),  # line 1
        )
        self._token_text = _token_texts(tokenizer)

    def _line_valid(self, line: str, line_no: int) -> bool:
        """Is *line* a valid prefix of a legal block line?"""
        prefix = _LINE_PREFIXES[line_no]
        if len(line) <= len(prefix):
            return prefix.startswith(line)
        if not line.startswith(prefix):
            return False
        rest = line[len(prefix) :]
        if rest == " ":
            return True
        if not rest.startswith(" "):
            return False
        done, partial = _split_list(rest[1:])
        idents = self._idents[line_no]
        if any(name not in idents for name in done):
            return False
        # A separator is typed one character at a time, so "customers," and
        # "customers, " are legal in-progress states with an empty partial.
        return partial == "" or any(i.startswith(partial) for i in idents)

    def _text_valid(self, text: str) -> bool:
        lines = text.split("\n")
        if len(lines) > 3:
            return True  # block over; SQL body is free
        for line_no, line in enumerate(lines[:2]):
            complete = line_no < len(lines) - 1
            if complete:
                # A finished line must end on a *complete* identifier. That name
                # lives in the partial slot (nothing follows it), so an exact
                # match is required there -- a dangling separator leaves the
                # partial empty and is correctly rejected.
                prefix = _LINE_PREFIXES[line_no]
                if line == prefix:
                    continue
                if not self._line_valid(line, line_no):
                    return False
                _done, partial = _split_list(line[len(prefix) + 1 :])
                if partial not in self._idents[line_no]:
                    return False
            elif not self._line_valid(line, line_no):
                return False
        return True

    def allowed_token_mask(
        self, prefix_token_ids: Sequence[int], vocab_size: int
    ) -> Sequence[bool]:
        generated = self._tokenizer.decode(list(prefix_token_ids[self._prompt_len :]))
        if generated.count("\n") >= 2:
            return [True] * vocab_size
        # Already off the manifold (the model skipped the block, or named an
        # identifier the schema lacks): no continuation can repair it, so
        # masking further would leave the decoder with no legal move.
        if generated and not self._text_valid(generated):
            return [True] * vocab_size
        mask = [False] * vocab_size
        for token_id in range(min(vocab_size, len(self._token_text))):
            token = self._token_text[token_id]
            if token and self._text_valid(generated + token):
                mask[token_id] = True
        return mask
