"""Block-constrained decoding: the linking block may only name real schema.

v3 models emit ``-- tables: .../-- columns: ...`` before the SQL. The block
conditions everything after it, so a hallucinated identifier there cascades
(exposure bias). This constraint masks generation *inside the block* to
tokens that keep every named identifier a real one -- the SQL body afterwards
is left completely free. Implements the ``SQLConstraint`` protocol pinned in
``eval/constrain.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from sqlpup.eval.blockmask import LinkingBlockConstraint

TOKENIZER_PATH = Path(__file__).parents[2] / "artifacts" / "tokenizer" / "tokenizer.json"

pytestmark = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(), reason="shipped tokenizer artifact not present"
)

SCHEMA = {
    "customers": ["id", "name", "city"],
    "orders": ["id", "customer_id", "total"],
}


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def _mask_for(tok: Tokenizer, constraint: LinkingBlockConstraint, text: str) -> list[bool]:
    ids = tok.encode(text).ids
    return list(constraint.allowed_token_mask(ids, tok.get_vocab_size()))


def _allows(tok: Tokenizer, mask: list[bool], continuation: str) -> bool:
    ids = tok.encode(continuation).ids
    return bool(ids) and mask[ids[0]]


def test_real_table_continuation_allowed_fake_disallowed(tok: Tokenizer) -> None:
    constraint = LinkingBlockConstraint(SCHEMA, tok, prompt_len=0)
    mask = _mask_for(tok, constraint, "-- tables: cust")
    assert _allows(tok, mask, "omers")  # customers is real
    # a continuation completing a non-schema identifier must be masked out
    assert not _allows(tok, mask, "ard")  # "custard" is not a table


def test_column_line_only_allows_qualified_schema_columns(tok: Tokenizer) -> None:
    constraint = LinkingBlockConstraint(SCHEMA, tok, prompt_len=0)
    mask = _mask_for(tok, constraint, "-- tables: orders\n-- columns: orders.")
    assert _allows(tok, mask, "total")
    assert not _allows(tok, mask, "zzz_fake")


def test_after_block_everything_is_free(tok: Tokenizer) -> None:
    constraint = LinkingBlockConstraint(SCHEMA, tok, prompt_len=0)
    mask = _mask_for(tok, constraint, "-- tables: orders\n-- columns: orders.total\nSELECT")
    assert all(mask)


def test_prompt_tokens_are_ignored(tok: Tokenizer) -> None:
    prompt = "CREATE TABLE nothere (x INT);\n-- SQL question etc\n"
    prompt_len = len(tok.encode(prompt).ids)
    constraint = LinkingBlockConstraint(SCHEMA, tok, prompt_len=prompt_len)
    mask = _mask_for(tok, constraint, prompt + "-- tables: ord")
    assert _allows(tok, mask, "ers")
    assert not _allows(tok, mask, "inary")  # "ordinary" is not a table


def test_newline_allowed_only_after_complete_identifier(tok: Tokenizer) -> None:
    constraint = LinkingBlockConstraint(SCHEMA, tok, prompt_len=0)
    incomplete = _mask_for(tok, constraint, "-- tables: cust")
    complete = _mask_for(tok, constraint, "-- tables: customers")
    assert not _allows(tok, incomplete, "\n")
    assert _allows(tok, complete, "\n")


def test_constraint_stops_constraining_once_the_text_cannot_be_a_block(tok: Tokenizer) -> None:
    """Off-manifold text (the model skipped the block, or named a table that
    does not exist) is unrecoverable: no continuation makes it valid. Keep
    masking and the decoder has no legal move, so stand down instead."""
    constraint = LinkingBlockConstraint(SCHEMA, tok, prompt_len=0)
    for off_manifold in ("SELECT", "-- tables: ghosts", " -- tables:"):
        mask = _mask_for(tok, constraint, off_manifold)
        assert any(mask), f"{off_manifold!r} left the decoder with no legal token"
        assert all(mask), f"{off_manifold!r} should decode unconstrained once invalid"


def test_mask_allows_continuing_to_a_second_table(tok: Tokenizer) -> None:
    """Measured bug (v3 step-30K): after a complete name the mask allowed only
    a newline, forcing every block to name exactly one table and wrecking
    multi-table queries. Separators are typed one character at a time, so the
    intermediate state `customers,` must stay valid."""
    constraint = LinkingBlockConstraint(SCHEMA, tok, prompt_len=0)
    after_first = _mask_for(tok, constraint, "-- tables: customers")
    assert _allows(tok, after_first, ","), "cannot start a second table"

    mid_separator = _mask_for(tok, constraint, "-- tables: customers,")
    assert _allows(tok, mid_separator, " orders")

    second_name = _mask_for(tok, constraint, "-- tables: customers, ord")
    assert _allows(tok, second_name, "ers")
    assert not _allows(tok, second_name, "inary")  # still constrained to real names


def test_multi_table_block_is_never_blocked_end_to_end(tok: Tokenizer) -> None:
    constraint = LinkingBlockConstraint(SCHEMA, tok, prompt_len=0)
    text = "-- tables: customers, orders\n-- columns: customers.id, orders.total\nSELECT 1"
    ids = tok.encode(text).ids
    for k in range(len(ids)):
        mask = constraint.allowed_token_mask(ids[:k], tok.get_vocab_size())
        assert mask[ids[k]], f"blocked at step {k}: {tok.decode(ids[:k])!r}"
