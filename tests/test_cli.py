import json
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pytest

from sqlpup.cli import main
from sqlpup.config import TokenizerConfig
from sqlpup.io import Document, read_documents, write_documents

LONG_A = "SELECT col_a FROM table_one WHERE col_a IS NOT NULL ORDER BY col_a;\n" * 5
LONG_B = "SELECT col_b FROM table_two WHERE col_b IS NOT NULL ORDER BY col_b;\n" * 5


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "raw.jsonl.zst"
    write_documents(
        path,
        [
            Document(id="a", text=LONG_A, source="unit"),
            Document(id="a2", text=LONG_A, source="unit"),
            Document(id="b", text=LONG_B, source="unit"),
            Document(id="tiny", text="SELECT 1;", source="unit"),
        ],
    )
    return path


def test_clean_then_dedup_pipeline(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cleaned = tmp_path / "clean.jsonl.zst"
    assert main(["data", "clean", "--in", str(corpus), "--out", str(cleaned)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["dropped_short"] == 1
    assert stats["written"] == 3

    deduped = tmp_path / "dedup.jsonl.zst"
    assert main(["data", "dedup", "--in", str(cleaned), "--out", str(deduped)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats == {"written": 2, "seen": 3, "unique": 2, "dropped": 1}


def test_clean_merges_multiple_inputs_preserving_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The sample corpus is assembled by cleaning all five per-source files into
    # one combined file; each surviving document keeps its own provenance.
    a = tmp_path / "fineweb.jsonl.zst"
    b = tmp_path / "starcoder.jsonl.zst"
    write_documents(
        a,
        [
            Document(id="fw-1", text=LONG_A, source="fineweb_edu"),
            Document(id="fw-2", text="short", source="fineweb_edu"),
        ],
    )
    write_documents(b, [Document(id="sc-1", text=LONG_B, source="starcoder_sql")])
    out = tmp_path / "corpus_sample.jsonl.zst"

    assert main(["data", "clean", "--in", str(a), str(b), "--out", str(out)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["written"] == 2  # the "short" doc is dropped
    assert stats["dropped_short"] == 1

    docs = list(read_documents(out))
    assert {d.id for d in docs} == {"fw-1", "sc-1"}
    assert {d.source for d in docs} == {"fineweb_edu", "starcoder_sql"}


NEAR_BASE = (
    "the quarterly revenue report aggregates sales figures across every regional "
    "office and summarizes them into a single consolidated dashboard that senior "
    "managers carefully review each monday morning before the weekly leadership "
    "meeting begins inside the main conference room on the fourth floor of the "
    "headquarters building downtown, where the finance team presents its updated "
    "forecasts and highlights any accounts that require immediate follow up from "
    "the wider sales organization this week without further delay or escalation."
)
NEAR_VARIANT = NEAR_BASE.replace("monday", "tuesday")
UNRELATED = (
    "a gentle evening breeze carried the sweet scent of blooming jasmine across "
    "the quiet cobbled courtyard while small children chased one another between "
    "the old stone fountains and their patient grandparents rested beneath a "
    "broad shady oak tree, trading familiar stories about the long summers of "
    "years past when the crowded village markets overflowed with ripe fruit, "
    "handmade crafts, and the cheerful music of travelling performers each weekend."
)


def test_dedup_near_drops_near_duplicate_after_exact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw.jsonl.zst"
    write_documents(
        raw,
        [
            Document(id="orig", text=NEAR_BASE, source="unit"),
            Document(id="near", text=NEAR_VARIANT, source="unit"),
            Document(id="other", text=UNRELATED, source="unit"),
        ],
    )
    out = tmp_path / "dedup.jsonl.zst"
    assert main(["data", "dedup", "--in", str(raw), "--out", str(out), "--near"]) == 0
    stats = json.loads(capsys.readouterr().out)
    # exact keeps all three (none is byte-identical); near drops the variant.
    assert stats["seen"] == 3
    assert stats["dropped"] == 0
    assert stats["near_duplicates_dropped"] == 1
    assert stats["too_short_passed"] == 0
    assert stats["written"] == 2
    assert [d.id for d in read_documents(out)] == ["orig", "other"]


def test_dedup_without_near_is_exact_only(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "dedup.jsonl.zst"
    assert main(["data", "dedup", "--in", str(corpus), "--out", str(out)]) == 0
    stats = json.loads(capsys.readouterr().out)
    # no near-dedup keys leak into the default (exact-only) stats line.
    assert "near_duplicates_dropped" not in stats
    assert set(stats) == {"written", "seen", "unique", "dropped"}


def test_decontaminate_command(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(json.dumps([{"question": "irrelevant question", "SQL": "SELECT nothing"}]))
    out = tmp_path / "decontam.jsonl.zst"
    code = main(
        [
            "data",
            "decontaminate",
            "--in",
            str(corpus),
            "--out",
            str(out),
            "--eval-json",
            str(eval_json),
            "--ngram",
            "5",
        ]
    )
    assert code == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["eval_texts"] == 1
    assert stats["written"] == stats["kept"] == 4


def test_decontaminate_with_prebuilt_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from sqlpup.data.decontaminate import NgramIndex

    eval_text = "how many students scored above ninety in the final math exam last spring semester"
    index_path = tmp_path / "eval_index.json"
    NgramIndex.build([eval_text], n=13).save(index_path)

    corpus = tmp_path / "raw.jsonl.zst"
    write_documents(
        corpus,
        [
            Document(id="bad", text=f"prefix noise {eval_text} suffix noise", source="unit"),
            Document(id="ok", text="SELECT 1;", source="unit"),
        ],
    )
    out = tmp_path / "decontam.jsonl.zst"
    code = main(
        [
            "data",
            "decontaminate",
            "--in",
            str(corpus),
            "--out",
            str(out),
            "--index-in",
            str(index_path),
        ]
    )
    assert code == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["written"] == stats["kept"] == 1
    assert stats["dropped"] == 1
    assert "eval_texts" not in stats  # nothing was rebuilt; a prebuilt index was loaded
    assert [d.id for d in read_documents(out)] == ["ok"]


def test_decontaminate_requires_exactly_one_index_source(tmp_path: Path) -> None:
    # neither --eval-json nor --index-in: argparse error
    with pytest.raises(SystemExit):
        main(["data", "decontaminate", "--in", "x", "--out", "y"])
    # both at once: mutually exclusive argparse error
    with pytest.raises(SystemExit):
        main(
            [
                "data",
                "decontaminate",
                "--in",
                "x",
                "--out",
                "y",
                "--eval-json",
                "a.json",
                "--index-in",
                "i.json",
            ]
        )


def test_build_eval_index_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile

    from sqlpup.data import download

    fixture = Path(__file__).parent / "data" / "fixtures" / "bird_dev_sample.json"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dev_20240627/dev.json", fixture.read_bytes())
    payload = buf.getvalue()

    def fake_download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)

    monkeypatch.setattr(download, "_download_artifact", fake_download)

    cfg = tmp_path / "eval_sets.yaml"
    cfg.write_text(
        "sources:\n"
        "  - name: bird_dev\n"
        "    url: http://eval.test/dev.zip\n"
        "    archive_member: dev_20240627/dev.json\n"
        "    fields: [question, SQL, evidence]\n"
        "    expected_examples: 4\n"
    )
    index_out = tmp_path / "artifacts" / "decontam" / "eval_index.json"
    code = main(
        [
            "data",
            "build-eval-index",
            "--config",
            str(cfg),
            "--data-dir",
            str(tmp_path / "eval"),
            "--index-out",
            str(index_out),
        ]
    )
    assert code == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["n"] == 13
    assert stats["sources"]["bird_dev"]["examples"] == 4
    assert index_out.exists()
    assert (index_out.parent / "eval_index_report.json").exists()


def test_data_stats_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus_dir = tmp_path / "corpus"
    write_documents(
        corpus_dir / "alpha.jsonl.zst",
        [
            Document(id="a1", text="x" * 100, source="alpha"),
            Document(id="a2", text="y" * 300, source="alpha"),
        ],
    )
    write_documents(corpus_dir / "beta.jsonl", [Document(id="b1", text="z" * 800, source="beta")])

    cfg = tmp_path / "mix.yaml"
    cfg.write_text(
        "seed: 1\n"
        f"out_dir: {corpus_dir}\n"
        "sources:\n"
        + "".join(
            f"  - name: {name}\n"
            "    hf_path: null\n"
            "    hf_name: null\n"
            "    split: train\n"
            "    license: MIT\n"
            f"    target_tokens: {target}\n"
            "    text_field: text\n"
            for name, target in (("alpha", 200), ("beta", 100), ("gamma", 500))
        )
    )

    report = tmp_path / "artifacts" / "stats.json"
    # --corpus-dir omitted on purpose: it must default to the mix out_dir.
    assert main(["data", "stats", "--config", str(cfg), "--report-out", str(report)]) == 0

    captured = capsys.readouterr()
    # stdout is a single JSON stats line (the emitted report); the human-readable
    # table goes to stderr so every stage keeps its one-JSON-line stdout contract.
    stdout_payload = json.loads(captured.out)
    table = captured.err
    assert all(name in table for name in ("alpha", "beta", "gamma"))
    assert "TOTAL" in table

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert stdout_payload == payload  # the stdout stats line matches the written report
    by = {s["name"]: s for s in payload["sources"]}
    assert by["alpha"]["approx_tokens"] == 100
    assert by["beta"]["approx_tokens"] == 200
    assert by["gamma"]["present"] is False
    assert payload["totals"]["approx_tokens"] == 300
    assert payload["approx_tokens_per_char"] == 0.25
    datetime.fromisoformat(payload["generated_at"])  # provenance timestamp parses


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    """Deterministic stand-in for a trained tokenizer: one token per document.

    Keeps the shard CLI tests hermetic and fast -- no BPE training or shipped
    artifact needed. Each document contributes exactly ``[1]`` + the EOS the
    writer appends (id 0), i.e. 2 tokens, so token accounting is trivially
    predictable and the holdout routing can be asserted from the counts.
    """

    def token_to_id(self, token: str) -> int | None:
        return 0 if token == "<|end|>" else None

    def encode(self, text: str) -> _FakeEncoding:
        return _FakeEncoding([1])


def test_data_shard_holdout_split_routes_by_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import xxhash

    from sqlpup.tokenizer import train as tok_train

    ids = [f"doc-{i}" for i in range(60)]
    corpus = tmp_path / "corpus.jsonl.zst"
    write_documents(corpus, [Document(id=i, text="SELECT 1;", source="unit") for i in ids])
    monkeypatch.setattr(tok_train, "load_tokenizer", lambda _p: _FakeTokenizer())

    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    code = main(
        [
            "data",
            "shard",
            "--in",
            str(corpus),
            "--tokenizer",
            "ignored.json",
            "--out-dir",
            str(train_dir),
            "--holdout-out-dir",
            str(eval_dir),
            "--holdout-mod",
            "5",
        ]
    )
    assert code == 0
    stats = json.loads(capsys.readouterr().out)

    held = {i for i in ids if xxhash.xxh64(i.encode()).intdigest() % 5 == 0}
    assert 0 < len(held) < len(ids)  # a real split actually happened
    # Non-holdout docs -> train, holdout docs -> eval; the two are disjoint.
    assert stats["docs"] == len(ids) - len(held)
    assert stats["holdout_docs"] == len(held)
    # Each doc is [1] + EOS = 2 tokens in whichever split it lands.
    assert stats["total_tokens"] == 2 * (len(ids) - len(held))
    assert stats["holdout_total_tokens"] == 2 * len(held)
    assert (train_dir / "index.json").exists()
    assert (eval_dir / "index.json").exists()


def test_data_shard_without_holdout_is_single_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlpup.tokenizer import train as tok_train

    corpus = tmp_path / "corpus.jsonl.zst"
    write_documents(
        corpus, [Document(id=f"d{i}", text="SELECT 1;", source="unit") for i in range(10)]
    )
    monkeypatch.setattr(tok_train, "load_tokenizer", lambda _p: _FakeTokenizer())

    code = main(
        ["data", "shard", "--in", str(corpus), "--tokenizer", "x.json", "--out-dir", str(tmp_path)]
    )
    assert code == 0
    stats = json.loads(capsys.readouterr().out)
    assert set(stats) == {"docs", "total_tokens", "shards"}  # no holdout keys leak
    assert stats["docs"] == 10


def test_data_shard_rejects_nonpositive_holdout_mod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlpup.tokenizer import train as tok_train

    corpus = tmp_path / "corpus.jsonl.zst"
    write_documents(corpus, [Document(id="a", text="SELECT 1;", source="unit")])
    monkeypatch.setattr(tok_train, "load_tokenizer", lambda _p: _FakeTokenizer())
    with pytest.raises(SystemExit):
        main(
            [
                "data",
                "shard",
                "--in",
                str(corpus),
                "--tokenizer",
                "x.json",
                "--out-dir",
                str(tmp_path / "train"),
                "--holdout-out-dir",
                str(tmp_path / "eval"),
                "--holdout-mod",
                "0",
            ]
        )


def test_tokenizer_train_holdout_excludes_reserved_docs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import xxhash

    from sqlpup.tokenizer import train as tok_train

    # Fixture docs whose text embeds the id, so we can recover which ids the
    # trainer actually saw from the captured corpus.
    ids = [f"doc-{i}" for i in range(60)]
    corpus_path = tmp_path / "corpus_sample.jsonl.zst"
    write_documents(
        corpus_path,
        [Document(id=i, text=f"{i} SELECT 1;", source="unit") for i in ids],
    )

    captured_texts: list[str] = []

    def fake_train_bpe(config: TokenizerConfig, corpus: Iterable[str]) -> Path:
        captured_texts.extend(corpus)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        path = config.out_dir / "tokenizer.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(tok_train, "train_bpe", fake_train_bpe)

    cfg = tmp_path / "tok.yaml"
    cfg.write_text(
        "vocab_size: 512\n"
        'special_tokens: ["<|end|>"]\n'
        "sample_docs: 1000\n"
        f"out_dir: {tmp_path / 'tok'}\n"
    )
    code = main(
        [
            "tokenizer",
            "train",
            "--config",
            str(cfg),
            "--corpus",
            str(corpus_path),
            "--holdout-mod",
            "5",
        ]
    )
    assert code == 0
    stats = json.loads(capsys.readouterr().out)

    expected_held = {i for i in ids if xxhash.xxh64(i.encode()).intdigest() % 5 == 0}
    seen_ids = {t.split()[0] for t in captured_texts}
    # Exactly the non-holdout ids reach the trainer; held-out ids are absent.
    assert seen_ids == set(ids) - expected_held
    assert seen_ids.isdisjoint(expected_held)
    assert stats["holdout_mod"] == 5
    assert stats["docs_held_out"] == len(expected_held)
    assert stats["docs_used"] == len(ids) - len(expected_held)


def test_tokenizer_train_without_holdout_uses_all_docs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlpup.tokenizer import train as tok_train

    ids = [f"doc-{i}" for i in range(20)]
    corpus_path = tmp_path / "corpus.jsonl.zst"
    write_documents(
        corpus_path, [Document(id=i, text=f"{i} SELECT 1;", source="unit") for i in ids]
    )

    captured_texts: list[str] = []

    def fake_train_bpe(config: TokenizerConfig, corpus: Iterable[str]) -> Path:
        captured_texts.extend(corpus)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        path = config.out_dir / "tokenizer.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(tok_train, "train_bpe", fake_train_bpe)

    cfg = tmp_path / "tok.yaml"
    cfg.write_text(
        "vocab_size: 512\n"
        'special_tokens: ["<|end|>"]\n'
        "sample_docs: 1000\n"
        f"out_dir: {tmp_path / 'tok'}\n"
    )
    assert main(["tokenizer", "train", "--config", str(cfg), "--corpus", str(corpus_path)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert len(captured_texts) == len(ids)
    assert "holdout_mod" not in stats  # off by default -> no holdout keys leak
    assert stats["docs_used"] == len(ids)


def test_tokenizer_train_rejects_nonpositive_holdout(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl.zst"
    write_documents(corpus_path, [Document(id="a", text="a SELECT 1;", source="unit")])
    cfg = tmp_path / "tok.yaml"
    cfg.write_text(
        "vocab_size: 512\n"
        'special_tokens: ["<|end|>"]\n'
        "sample_docs: 1000\n"
        f"out_dir: {tmp_path / 'tok'}\n"
    )
    with pytest.raises(SystemExit):
        main(
            [
                "tokenizer",
                "train",
                "--config",
                str(cfg),
                "--corpus",
                str(corpus_path),
                "--holdout-mod",
                "0",
            ]
        )


def test_cli_help_via_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sqlpup.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "sqlpup" in result.stdout


def test_train_without_torch_prints_friendly_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Simulate the optional `train` extra being absent by making the lazy import
    # raise, regardless of whether torch is actually installed in this env.
    from sqlpup import cli

    def _no_torch() -> object:
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(cli, "_load_trainer", _no_torch)
    code = cli.main(["train", "--config", "nonexistent.yaml"])
    assert code != 0
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing leaks onto the stats stream
    assert "sqlpup[train]" in captured.err  # names the actionable extra
    assert "uv sync --extra train" in captured.err
    assert "Traceback" not in captured.err  # friendly message, not a stack trace


def _one_hf_source_mix(config: Path, out_dir: Path) -> None:
    """Write a one-source hf mix config (the-stack-dedup shape) at ``config``."""
    config.write_text(
        "seed: 1\n"
        f"out_dir: {out_dir}\n"
        "sources:\n"
        "  - name: stack_python\n"
        "    hf_path: bigcode/the-stack-dedup\n"
        "    hf_name: null\n"
        "    split: train\n"
        "    license: permissive\n"
        "    target_tokens: 100000\n"
        "    text_field: content\n"
        "    hf_data_dir: data/python\n",
        encoding="utf-8",
    )


def test_data_download_fast_exits_after_writing_on_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # `sqlpup data download` streams parquet sources (e.g. bigcode/the-stack-dedup)
    # through pyarrow, whose process-global thread pool intermittently deadlocks in
    # its C++ static destructor at interpreter shutdown: the command writes every
    # file and manifest and prints its summary, then hangs. The fix skips that
    # teardown with a process-level fast exit on the SUCCESS path only. Patch the
    # hook to record (never os._exit, which would kill pytest) and assert it fires
    # once, after the summary is emitted and the manifest is durably on disk.
    import datasets

    from sqlpup import cli

    out_dir = tmp_path / "out"
    config = tmp_path / "mix.yaml"
    _one_hf_source_mix(config, out_dir)

    # Offline: a fake HF stream with more records than the doc cap, so the reader
    # stops early exactly as it does mid-parquet on the network -- no network.
    def fake_load_dataset(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [{"content": f"SELECT {i};"} for i in range(50)]

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    manifest = out_dir / "stack_python.manifest.json"
    fast_exit_saw_manifest: list[bool] = []

    def fake_fast_exit() -> None:
        fast_exit_saw_manifest.append(manifest.exists())

    monkeypatch.setattr(cli, "_download_fast_exit", fake_fast_exit)

    code = cli.main(["data", "download", "--config", str(config), "--limit-docs", "5"])

    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["downloads"][0]["docs"] == 5  # stopped at the cap, not all 50 records
    assert json.loads(manifest.read_text(encoding="utf-8"))["complete"] is True
    # Fired exactly once, on the success path, AFTER the manifest was on disk.
    assert fast_exit_saw_manifest == [True]


def test_data_download_does_not_fast_exit_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fast exit is a success-path optimization: a streaming error must
    # propagate with the normal non-zero exit semantics, never be masked by a
    # premature exit(0) that would hide the failure.
    import datasets

    from sqlpup import cli

    config = tmp_path / "mix.yaml"
    _one_hf_source_mix(config, tmp_path / "out")

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("stream blew up")

    monkeypatch.setattr(datasets, "load_dataset", boom)

    fired: list[str] = []
    monkeypatch.setattr(cli, "_download_fast_exit", lambda: fired.append("x"))

    with pytest.raises(RuntimeError, match="stream blew up"):
        cli.main(["data", "download", "--config", str(config), "--limit-docs", "5"])
    assert fired == []  # never reached the success-path fast exit
