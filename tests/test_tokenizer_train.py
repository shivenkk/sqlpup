import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
import xxhash

from sqlpup.config import TokenizerConfig
from sqlpup.tokenizer.train import (
    build_pre_tokenizer,
    is_holdout,
    load_tokenizer,
    train_bpe,
)

CORPUS = [
    "SELECT name, age FROM users WHERE age > 21 ORDER BY name;",
    "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);",
    "INSERT INTO users (name, age) VALUES ('ada', 36);",
] * 80


def _config(tmp_path: Path) -> TokenizerConfig:
    return TokenizerConfig(
        vocab_size=512,
        special_tokens=("<|schema|>", "<|question|>", "<|think|>", "<|sql|>", "<|end|>"),
        sample_docs=1_000,
        out_dir=tmp_path / "tok",
    )


def test_trains_and_saves(tmp_path: Path) -> None:
    path = train_bpe(_config(tmp_path), CORPUS)
    assert path.exists()
    tokenizer = load_tokenizer(path)
    assert tokenizer.get_vocab_size() <= 512


def test_special_tokens_have_ids_and_stay_atomic(tmp_path: Path) -> None:
    tokenizer = load_tokenizer(train_bpe(_config(tmp_path), CORPUS))
    sql_id = tokenizer.token_to_id("<|sql|>")
    assert sql_id is not None
    ids = tokenizer.encode("<|sql|>\nSELECT 1;").ids
    assert ids[0] == sql_id


def test_encode_decode_roundtrip(tmp_path: Path) -> None:
    tokenizer = load_tokenizer(train_bpe(_config(tmp_path), CORPUS))
    text = "SELECT name FROM users WHERE age > 21;"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_trains_from_lazy_generator(tmp_path: Path) -> None:
    # The CLI streams the corpus as a generator instead of materializing it, so
    # train_bpe must accept a one-shot iterator, not just a list.
    def gen() -> Iterator[str]:
        yield from CORPUS

    path = train_bpe(_config(tmp_path), gen())
    assert path.exists()


def test_is_holdout_matches_documented_formula_and_is_deterministic() -> None:
    # The rule is exactly `xxh64(id) % N == 0`, computed against the raw xxhash
    # so a regression in the wrapper is caught, and it is stable across calls.
    ids = [f"synsql-{i}" for i in range(200)] + ["fineweb_edu-0", "stack_python-99", ""]
    for mod in (2, 3, 50):
        for doc_id in ids:
            expected = xxhash.xxh64(doc_id.encode()).intdigest() % mod == 0
            assert is_holdout(doc_id, mod) is expected
            assert is_holdout(doc_id, mod) is is_holdout(doc_id, mod)


def test_is_holdout_reserves_roughly_one_over_n() -> None:
    # Sanity on the partition size: with N=50 the held-out slice is a small,
    # non-trivial fraction of a few thousand ids (not empty, not everything).
    ids = [f"doc-{i}" for i in range(5_000)]
    held = [i for i in ids if is_holdout(i, 50)]
    assert 0 < len(held) < len(ids)
    assert 50 < len(held) < 150  # ~100 expected (5000/50), allowing hash spread


# --- configurable pre-tokenizer ---------------------------------------------

TRICKY = [
    "SELECT * FROM t WHERE id IN (12345, 678, 9) AND ts >= 20190102 LIMIT 1000;",
    "grüße 日本語 '\U0001d54f\U0001f680' -- café comment\n\tINSERT INTO x VALUES ('a''b');",
]


def _code_config(tmp_path: Path) -> TokenizerConfig:
    return dataclasses.replace(_config(tmp_path), pre_tokenizer="code")


def test_default_config_uses_byte_level_pre_tokenizer(tmp_path: Path) -> None:
    assert _config(tmp_path).pre_tokenizer == "byte_level"


def test_code_pre_tokenizer_trains_and_roundtrips_exactly(tmp_path: Path) -> None:
    # Byte-level mapping is last in the "code" sequence, so the trained tokenizer
    # must still round-trip arbitrary text (unicode, doubled quotes, tabs) exactly.
    tokenizer = load_tokenizer(train_bpe(_code_config(tmp_path), CORPUS))
    for text in TRICKY:
        assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_code_pre_tokenizer_keeps_special_tokens_atomic(tmp_path: Path) -> None:
    tokenizer = load_tokenizer(train_bpe(_code_config(tmp_path), CORPUS))
    ids = tokenizer.encode("<|sql|>\nSELECT 1;").ids
    assert ids[0] == tokenizer.token_to_id("<|sql|>")


def test_build_pre_tokenizer_code_chunks_long_digits_default_keeps_one_run() -> None:
    # The digit lever: the "code" variant chunks a long numeral into <=3-digit
    # pieces (cl100k-style), while the byte-level default keeps one all-digit run.
    code = [piece for piece, _ in build_pre_tokenizer("code").pre_tokenize_str("1234567")]
    byte_level = [
        piece for piece, _ in build_pre_tokenizer("byte_level").pre_tokenize_str("1234567")
    ]
    assert code == ["123", "456", "7"]
    assert byte_level == ["1234567"]


def test_build_pre_tokenizer_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="pre_tokenizer"):
        build_pre_tokenizer("nope")
