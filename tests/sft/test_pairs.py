"""SFT pair building: byte-identical prompts, boundary-safe masking, ctx filter.

These tests run against the REAL shipped tokenizer artifact -- the whole point
of the boundary test is to catch byte-level BPE merging across the
prompt/completion seam, which no toy tokenizer would reproduce.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from sqlpup.eval.prompts import BIRD_DDL_V1, schema_ddl
from sqlpup.sft.pairs import IGNORE_INDEX, BoundaryError, SFTPair, build_pair

TOKENIZER_PATH = Path(__file__).parents[2] / "artifacts" / "tokenizer" / "tokenizer.json"

pytestmark = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(), reason="shipped tokenizer artifact not present"
)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_PATH))


@pytest.fixture
def fixture_ddl(tmp_path: Path) -> str:
    path = tmp_path / "f.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE t (id INTEGER, name TEXT);")
    con.commit()
    con.close()
    return schema_ddl(path)


def _eos_id(tokenizer: Tokenizer) -> int:
    eos = tokenizer.token_to_id("<|end|>")
    assert eos is not None
    return int(eos)


def test_build_pair_masks_exactly_the_prompt_prefix(tokenizer: Tokenizer, fixture_ddl: str) -> None:
    pair = build_pair(
        tokenizer,
        ddl=fixture_ddl,
        question="How many rows does t have?",
        evidence="rows refers to COUNT(*)",
        sql="SELECT COUNT(*) FROM t",
        eos_id=_eos_id(tokenizer),
    )
    assert isinstance(pair, SFTPair)
    assert len(pair.input_ids) == len(pair.labels) == pair.total_tokens
    assert 0 < pair.prompt_tokens < pair.total_tokens
    # Prompt positions carry no loss; completion positions supervise their token.
    assert all(label == IGNORE_INDEX for label in pair.labels[: pair.prompt_tokens])
    completion_labels = pair.labels[pair.prompt_tokens :]
    assert all(label != IGNORE_INDEX for label in completion_labels)
    assert list(completion_labels) == list(pair.input_ids[pair.prompt_tokens :])
    # The supervised text round-trips to the SQL + terminator, ending in eos.
    assert pair.input_ids[-1] == _eos_id(tokenizer)
    decoded = tokenizer.decode(list(pair.input_ids[pair.prompt_tokens :]))
    assert decoded.strip() == "SELECT COUNT(*) FROM t;"


def test_prompt_renders_byte_identically_to_eval(tokenizer: Tokenizer, fixture_ddl: str) -> None:
    question, evidence = "names?", ""
    pair = build_pair(
        tokenizer,
        ddl=fixture_ddl,
        question=question,
        evidence=evidence,
        sql="SELECT name FROM t",
        eos_id=_eos_id(tokenizer),
    )
    prompt = BIRD_DDL_V1.render(fixture_ddl, question=question, evidence=evidence)
    assert list(pair.input_ids[: pair.prompt_tokens]) == tokenizer.encode(prompt).ids


def test_boundary_stability_holds_across_realistic_sql_shapes(
    tokenizer: Tokenizer, fixture_ddl: str
) -> None:
    # If byte-level BPE merged across the prompt/completion seam for any of
    # these, build_pair would raise BoundaryError and this test would fail.
    sqls = [
        "SELECT COUNT(*) FROM t",
        "select id , name from t where id > 1",
        "SELECT T1.name FROM t AS T1 WHERE T1.id = (SELECT MAX(id) FROM t)",
        "WITH c AS (SELECT id FROM t) SELECT * FROM c",
    ]
    for sql in sqls:
        pair = build_pair(
            tokenizer,
            ddl=fixture_ddl,
            question="q?",
            evidence="",
            sql=sql,
            eos_id=_eos_id(tokenizer),
        )
        assert not pair.overflow


def test_overflow_is_flagged_not_truncated(tokenizer: Tokenizer, fixture_ddl: str) -> None:
    pair = build_pair(
        tokenizer,
        ddl=fixture_ddl,
        question="q?" * 4000,  # blows far past the context on its own
        evidence="",
        sql="SELECT 1",
        eos_id=_eos_id(tokenizer),
        context_limit=2048,
    )
    assert pair.overflow
    assert pair.total_tokens > 2048  # kept intact for the caller to filter/count


def test_boundary_violation_raises_loudly(tokenizer: Tokenizer, fixture_ddl: str) -> None:
    # Force a seam merge: a completion that starts mid-word (no leading
    # whitespace/newline) glues onto the prompt's last token under byte BPE.
    with pytest.raises(BoundaryError):
        build_pair(
            tokenizer,
            ddl=fixture_ddl,
            question="q?",
            evidence="",
            sql="SELECT 1",
            eos_id=_eos_id(tokenizer),
            _prompt_override="prefix-with-no-trailing-newline",
            _completion_override="continuation",
        )
