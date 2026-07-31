"""Generator protocol, completion->SQL extraction, and the scripted test fake."""

from __future__ import annotations

import pytest

from sqlpup.eval.generate import FakeGenerator, SQLGenerator, extract_sql


def test_extract_sql_cuts_at_first_semicolon_and_strips() -> None:
    assert extract_sql("  SELECT 1;\nSELECT 2;") == "SELECT 1"


def test_extract_sql_without_semicolon_returns_stripped_text() -> None:
    assert extract_sql("SELECT a FROM t\n") == "SELECT a FROM t"


def test_extract_sql_unwraps_a_code_fence() -> None:
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"


def test_extract_sql_of_empty_text_is_empty() -> None:
    assert extract_sql("   \n") == ""


def test_fake_generator_pops_script_and_records_calls() -> None:
    fake = FakeGenerator(script=[["SELECT 1"], ["SELECT 2"]])
    assert fake.generate(["p1"]) == ["SELECT 1"]
    assert fake.generate(["p2"]) == ["SELECT 2"]
    assert fake.calls == [["p1"], ["p2"]]


def test_fake_generator_rejects_mismatched_round_width() -> None:
    fake = FakeGenerator(script=[["only-one"]])
    with pytest.raises(AssertionError):
        fake.generate(["p1", "p2"])


def test_fake_generator_satisfies_the_protocol() -> None:
    generator: SQLGenerator = FakeGenerator(script=[])
    assert hasattr(generator, "generate")


def test_constraints_are_explicitly_not_implemented_yet() -> None:
    class _NullConstraint:
        def allowed_token_mask(self, prefix_token_ids: object, vocab_size: int) -> list[bool]:
            return [True] * vocab_size

    with pytest.raises(NotImplementedError):
        FakeGenerator(script=[], constraint=_NullConstraint())


def test_extract_sql_skips_leading_comment_lines() -> None:
    completion = (
        "-- tables: customers, orders\n"
        "-- columns: customers.id, orders.customer_id\n"
        "SELECT COUNT(*) FROM orders;"
    )
    assert extract_sql(completion) == "SELECT COUNT(*) FROM orders"


def test_extract_sql_all_comments_yields_empty() -> None:
    assert extract_sql("-- tables: a\n-- columns: a.x\n") == ""


def test_extract_sql_keeps_inline_comment_after_statement_start() -> None:
    # Only *leading* comment lines are the linking block; a mid-query comment
    # is part of the SQL and sqlite handles it.
    assert extract_sql("SELECT 1 -- why not\n") == "SELECT 1 -- why not"
