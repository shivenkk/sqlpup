"""SFT prepare: stream huge JSON arrays, build filtered pairs, write manifest."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from sqlpup.sft.prepare import iter_json_array, prepare_pairs

TOKENIZER_PATH = Path(__file__).parents[2] / "artifacts" / "tokenizer" / "tokenizer.json"

pytestmark = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(), reason="shipped tokenizer artifact not present"
)


def test_iter_json_array_streams_objects_without_loading_whole_file(tmp_path: Path) -> None:
    rows = [{"i": i, "s": "x" * 50} for i in range(200)]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    seen = list(iter_json_array(path, buffer_bytes=256))  # buffer far smaller than the file
    assert len(seen) == 200
    assert seen[0] == {"i": 0, "s": "x" * 50}
    assert seen[199]["i"] == 199


def test_iter_json_array_handles_nested_braces_and_strings(tmp_path: Path) -> None:
    rows = [{"sql": "SELECT \"}\" FROM t WHERE x = '{'", "d": {"k": [1, {"z": "}"}]}}]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert list(iter_json_array(path)) == rows


def _write_db(root: Path, db_id: str) -> None:
    d = root / db_id
    d.mkdir(parents=True)
    con = sqlite3.connect(d / f"{db_id}.sqlite")
    con.executescript("CREATE TABLE t (id INTEGER, name TEXT);")
    con.commit()
    con.close()


def test_prepare_pairs_writes_pairs_and_manifest(tmp_path: Path) -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    db_root = tmp_path / "dbs"
    _write_db(db_root, "toy")
    rows = [
        {
            "db_id": "toy",
            "question": "how many?",
            "external_knowledge": "",
            "sql": "SELECT COUNT(*) FROM t",
        },
        {
            "db_id": "toy",
            "question": "names?",
            "external_knowledge": "n means name",
            "sql": "SELECT name FROM t",
        },
        {"db_id": "missing_db", "question": "q", "external_knowledge": "", "sql": "SELECT 1"},
        {"db_id": "toy", "question": "big?" * 4000, "external_knowledge": "", "sql": "SELECT 1"},
    ]
    out = tmp_path / "pairs.jsonl"
    manifest = prepare_pairs(rows, db_root, tokenizer, out, eos_id=6, context_limit=2048)

    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(written) == 2  # missing-db row skipped, overflow row filtered
    assert set(written[0]) == {"input_ids", "labels", "prompt_tokens"}
    assert len(written[0]["input_ids"]) == len(written[0]["labels"])
    assert manifest["written"] == 2
    assert manifest["skipped_missing_db"] == 1
    assert manifest["filtered_overflow"] == 1
    assert manifest["boundary_errors"] == 0
    assert manifest["prompt_spec"].startswith("bird-ddl-v1@")


def test_prepare_pairs_subsamples_deterministically(tmp_path: Path) -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    db_root = tmp_path / "dbs"
    _write_db(db_root, "toy")
    rows = [
        {"db_id": "toy", "question": f"q{i}", "external_knowledge": "", "sql": "SELECT 1"}
        for i in range(100)
    ]
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    m_a = prepare_pairs(rows, db_root, tokenizer, out_a, eos_id=6, sample_rate=0.3, seed=7)
    m_b = prepare_pairs(rows, db_root, tokenizer, out_b, eos_id=6, sample_rate=0.3, seed=7)
    assert m_a["written"] == m_b["written"]
    assert 10 <= m_a["written"] <= 50  # ~30 of 100
    assert out_a.read_text() == out_b.read_text()  # same seed -> identical output


def test_sft_prepare_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from sqlpup.cli import main

    db_root = tmp_path / "dbs"
    _write_db(db_root, "toy")
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(
            [{"db_id": "toy", "question": "q?", "external_knowledge": "", "sql": "SELECT 1"}]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "pairs.jsonl"
    rc = main(
        [
            "sft",
            "prepare",
            "--data",
            str(data),
            "--db-root",
            str(db_root),
            "--tokenizer",
            str(TOKENIZER_PATH),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written"] == 1
    assert out.exists()


def test_prepare_normalizes_bird_and_spider_row_fields(tmp_path: Path) -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    db_root = tmp_path / "dbs"
    _write_db(db_root, "toy")
    rows = [
        # BIRD-train style: SQL + evidence keys
        {"db_id": "toy", "question": "q1", "evidence": "hint", "SQL": "SELECT 1"},
        # Spider style: query key, no evidence
        {"db_id": "toy", "question": "q2", "query": "SELECT 2"},
        # SynSQL style (already supported)
        {"db_id": "toy", "question": "q3", "external_knowledge": "", "sql": "SELECT 3"},
    ]
    out = tmp_path / "mixed.jsonl"
    manifest = prepare_pairs(rows, db_root, tokenizer, out, eos_id=6)
    assert manifest["written"] == 3
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(written) == 3


def test_prepare_link_targets_prefixes_completion_with_linking_block(tmp_path: Path) -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    db_root = tmp_path / "dbs"
    _write_db(db_root, "toy")
    rows = [
        {
            "db_id": "toy",
            "question": "names?",
            "external_knowledge": "",
            "sql": "SELECT name FROM t",
        }
    ]
    out = tmp_path / "linked.jsonl"
    manifest = prepare_pairs(rows, db_root, tokenizer, out, eos_id=6, link_targets=True)
    assert manifest["written"] == 1
    assert manifest["targets"] == "linked-v1"
    pair = json.loads(out.read_text().splitlines()[0])
    completion_ids = [
        tok for tok, label in zip(pair["input_ids"], pair["labels"], strict=True) if label != -100
    ]
    completion = tokenizer.decode(completion_ids)
    assert completion.startswith("-- tables: t\n-- columns: t.name\n")
    assert "SELECT name FROM t" in completion


def test_prepare_reduce_overflow_recovers_oversized_pairs(tmp_path: Path) -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    db_root = tmp_path / "dbs"
    d = db_root / "wide"
    d.mkdir(parents=True)
    con = sqlite3.connect(d / "wide.sqlite")
    wide_cols = ", ".join(f"filler_column_{i} TEXT" for i in range(120))
    con.executescript(
        f"CREATE TABLE facts (id INTEGER PRIMARY KEY, amount REAL, {wide_cols});"
        "CREATE TABLE unrelated (id INTEGER PRIMARY KEY, blob TEXT);"
    )
    con.commit()
    con.close()
    rows = [
        {
            "db_id": "wide",
            "question": "total?",
            "external_knowledge": "",
            "sql": "SELECT SUM(amount) FROM facts",
        }
    ]
    strict = prepare_pairs(
        rows, db_root, tokenizer, tmp_path / "strict.jsonl", eos_id=6, context_limit=384
    )
    assert strict["written"] == 0 and strict["filtered_overflow"] == 1  # v2e behaviour
    recovered = prepare_pairs(
        rows,
        db_root,
        tokenizer,
        tmp_path / "rec.jsonl",
        eos_id=6,
        context_limit=384,
        reduce_overflow=True,
    )
    assert recovered["written"] == 1
    assert recovered["filtered_overflow"] == 0
    assert recovered["reduced_level1"] + recovered["reduced_level2"] == 1


def test_parallel_prepare_is_byte_identical_to_serial(tmp_path: Path) -> None:
    """Workers change speed, never output: same rows -> same file, same manifest."""
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    db_root = tmp_path / "dbs"
    _write_db(db_root, "toy")
    rows = [
        {
            "db_id": "toy",
            "question": f"question number {i}?",
            "external_knowledge": "n means name" if i % 3 else "",
            "sql": "SELECT name FROM t" if i % 2 else "SELECT COUNT(*) FROM t",
        }
        for i in range(60)
    ]
    serial_out = tmp_path / "serial.jsonl"
    parallel_out = tmp_path / "parallel.jsonl"
    serial = prepare_pairs(
        rows, db_root, tokenizer, serial_out, eos_id=6, link_targets=True, reduce_overflow=True
    )
    parallel = prepare_pairs(
        rows,
        db_root,
        tokenizer,
        parallel_out,
        eos_id=6,
        link_targets=True,
        reduce_overflow=True,
        workers=2,
        tokenizer_path=str(TOKENIZER_PATH),
    )
    assert parallel_out.read_text() == serial_out.read_text()
    ignored = {"out", "workers"}  # provenance fields that legitimately differ
    assert {k: v for k, v in parallel.items() if k not in ignored} == {
        k: v for k, v in serial.items() if k not in ignored
    }
    assert serial["workers"] == 1 and parallel["workers"] == 2
    assert serial["written"] == 60


def test_parallel_prepare_requires_a_tokenizer_path(tmp_path: Path) -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    db_root = tmp_path / "dbs"
    _write_db(db_root, "toy")
    with pytest.raises(ValueError, match="tokenizer_path"):
        prepare_pairs(
            [{"db_id": "toy", "question": "q", "external_knowledge": "", "sql": "SELECT 1"}],
            db_root,
            tokenizer,
            tmp_path / "x.jsonl",
            eos_id=6,
            workers=4,
        )


def test_prepare_cli_accepts_a_foreign_eos_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Qwen control arm uses its own tokenizer, whose EOS is not <|end|>.

    Hardcoding our tokenizer's sentinel would make the matched-recipe
    comparison impossible to run, so the token is a parameter.
    """
    from sqlpup.cli import main

    db_root = tmp_path / "dbs"
    _write_db(db_root, "toy")
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(
            [{"db_id": "toy", "question": "q?", "external_knowledge": "", "sql": "SELECT 1"}]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "pairs.jsonl"
    # a token our tokenizer really has, standing in for a foreign one
    rc = main(
        [
            "sft",
            "prepare",
            "--data",
            str(data),
            "--db-root",
            str(db_root),
            "--tokenizer",
            str(TOKENIZER_PATH),
            "--out",
            str(out),
            "--eos-token",
            "<|pad|>",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written"] == 1


def test_prepare_cli_fails_loudly_on_an_unknown_eos_token(tmp_path: Path) -> None:
    from sqlpup.cli import main

    db_root = tmp_path / "dbs"
    _write_db(db_root, "toy")
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps([{"db_id": "toy", "question": "q", "sql": "SELECT 1"}]), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="no-such-token"):
        main(
            [
                "sft",
                "prepare",
                "--data",
                str(data),
                "--db-root",
                str(db_root),
                "--tokenizer",
                str(TOKENIZER_PATH),
                "--out",
                str(tmp_path / "x.jsonl"),
                "--eos-token",
                "<|no-such-token|>",
            ]
        )


def test_train_cli_resolves_pad_for_a_foreign_tokenizer(tmp_path: Path) -> None:
    """The Qwen control arm died here: pad was hardcoded to our own sentinels,
    so `int(None)` threw after an hour of data prep. A tokenizer that has
    neither <|pad|> nor <|end|> must still train, falling back to its own EOS.
    """
    from sqlpup.cli import _resolve_pad_id

    class _Tok:
        def __init__(self, known: dict[str, int]) -> None:
            self.known = known

        def token_to_id(self, tok: str) -> int | None:
            return self.known.get(tok)

    # ours: a real pad token
    assert _resolve_pad_id(_Tok({"<|pad|>": 5, "<|end|>": 6}), "<|end|>") == 5
    # ours without pad: falls back to the eos we were told about
    assert _resolve_pad_id(_Tok({"<|end|>": 6}), "<|end|>") == 6
    # a foreign tokenizer (Qwen): neither of our sentinels, but its own eos
    assert _resolve_pad_id(_Tok({"<|endoftext|>": 151643}), "<|endoftext|>") == 151643


def test_train_cli_pad_resolution_fails_loudly_when_nothing_matches() -> None:
    from sqlpup.cli import _resolve_pad_id

    class _Empty:
        def token_to_id(self, tok):  # type: ignore[no-untyped-def]
            return None

    with pytest.raises(SystemExit, match="pad"):
        _resolve_pad_id(_Empty(), "<|nope|>")
