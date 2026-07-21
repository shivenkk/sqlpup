"""Offline tests for the tokenizer fertility comparison.

Every test here is fully offline: no tiktoken BPE downloads, no Hugging Face
downloads. The metric core is exercised with fake adapters over hand-computed
fixtures; the real reference tokenizers and the network eval loaders are only
touched by the separate CLI run, never here.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from sqlpup.config import TokenizerConfig
from sqlpup.io import Document, write_documents
from sqlpup.tokenizer import fertility
from sqlpup.tokenizer.fertility import (
    EvalGroupSpec,
    FertilityReport,
    GroupResult,
    TextGroup,
    TokenizerInfo,
    aggregate_results,
    build_eval_groups,
    holdout_groups,
    measure_group,
    render_markdown,
)
from sqlpup.tokenizer.train import is_holdout

FIXTURES = Path(__file__).parent / "data" / "fixtures"
BIRD_FIXTURE = FIXTURES / "bird_dev_sample.json"
SPIDER_FIXTURE = FIXTURES / "spider_dev_sample.json"


# --- fake adapters ----------------------------------------------------------


class CharAdapter:
    """One token per Unicode character (deterministic, offline)."""

    def __init__(self, name: str, vocab_size: int) -> None:
        self.name = name
        self.vocab_size = vocab_size

    def encode(self, text: str) -> int:
        return len(text)


class WordAdapter:
    """``k`` tokens per whitespace-split word (deterministic, offline)."""

    def __init__(self, name: str, vocab_size: int, k: int) -> None:
        self.name = name
        self.vocab_size = vocab_size
        self._k = k

    def encode(self, text: str) -> int:
        return self._k * len(text.split())


# --- metric core ------------------------------------------------------------


def test_measure_group_counts_bytes_words_and_tokens() -> None:
    group = TextGroup("g", ["ab cd", "xyz"])
    char = CharAdapter("char", 999)
    word = WordAdapter("word", 5, k=2)
    result = measure_group(group, [char, word])

    assert result.name == "g"
    assert result.docs == 2
    assert result.n_bytes == 8  # 5 + 3 ASCII bytes
    assert result.words == 3  # 2 + 1
    assert result.tokens == {"char": 8, "word": 6}  # 5+3 chars, 2*(2+1) words

    assert result.tokens_per_word("char") == pytest.approx(8 / 3)
    assert result.bytes_per_token("char") == pytest.approx(8 / 8)
    assert result.tokens_per_word("word") == pytest.approx(6 / 3)
    assert result.bytes_per_token("word") == pytest.approx(8 / 6)


def test_measure_group_bytes_are_utf8_not_char_count() -> None:
    # "áé" is 2 characters but 4 UTF-8 bytes: bytes must not be a char count.
    result = measure_group(TextGroup("u", ["áé"]), [CharAdapter("char", 1)])
    assert result.n_bytes == 4
    assert result.words == 1
    assert result.tokens == {"char": 2}


def test_measure_group_column_order_follows_adapter_order() -> None:
    a = CharAdapter("first", 1)
    b = WordAdapter("second", 2, k=1)
    assert list(measure_group(TextGroup("g", ["a b"]), [a, b]).tokens) == ["first", "second"]
    assert list(measure_group(TextGroup("g", ["a b"]), [b, a]).tokens) == ["second", "first"]


def test_aggregate_sums_totals_and_recomputes_ratios_not_averages() -> None:
    # Two groups with very different per-group ratios; the aggregate ratio must
    # come from summed totals, not from averaging the two ratios.
    g1 = GroupResult("g1", docs=1, n_bytes=10, words=2, tokens={"t": 4})
    g2 = GroupResult("g2", docs=3, n_bytes=90, words=8, tokens={"t": 16})
    overall = aggregate_results([g1, g2], name="overall")

    assert overall.name == "overall"
    assert overall.docs == 4
    assert overall.n_bytes == 100
    assert overall.words == 10
    assert overall.tokens == {"t": 20}
    assert overall.tokens_per_word("t") == pytest.approx(20 / 10)  # not (2+2)/2
    assert overall.bytes_per_token("t") == pytest.approx(100 / 20)


def test_ratios_guard_against_empty_groups() -> None:
    empty = GroupResult("e", docs=0, n_bytes=0, words=0, tokens={"t": 0})
    assert empty.tokens_per_word("t") == 0.0
    assert empty.bytes_per_token("t") == 0.0


# --- holdout corpus selector ------------------------------------------------


def test_holdout_groups_selects_only_holdout_docs_grouped_and_sorted(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl.zst"
    ids = [f"doc-{i}" for i in range(120)]
    sources = ["starcoder_sql", "fineweb_edu", "synsql"]
    docs = [
        Document(id=doc_id, text=f"{doc_id} SELECT 1 FROM t;", source=sources[i % len(sources)])
        for i, doc_id in enumerate(ids)
    ]
    write_documents(corpus, docs)

    groups = holdout_groups(corpus, holdout_mod=3)

    # groups are sorted by source name for a deterministic column/row order
    assert [g.name for g in groups] == sorted({d.source for d in docs})

    # exactly the is_holdout(id, 3) documents appear, grouped by their source
    expected: dict[str, set[str]] = {}
    for d in docs:
        if is_holdout(d.id, 3):
            expected.setdefault(d.source, set()).add(d.id)
    for group in groups:
        seen = {text.split()[0] for text in group.texts}
        assert seen == expected[group.name]
        assert all(is_holdout(doc_id, 3) for doc_id in seen)

    # the selector is a pure function of the file: same call, same groups
    again = holdout_groups(corpus, holdout_mod=3)
    assert [(g.name, list(g.texts)) for g in groups] == [(g.name, list(g.texts)) for g in again]


# --- eval-set field mapping (offline via fixture zips) ----------------------


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


BIRD_URL = "http://eval.test/bird_dev.zip"
SPIDER_URL = "http://eval.test/spider_data.zip"


def _write_eval_config(path: Path) -> None:
    path.write_text(
        "sources:\n"
        "  - name: bird_dev\n"
        f"    url: {BIRD_URL}\n"
        "    archive_member: dev_20240627/dev.json\n"
        "    fields: [question, SQL, evidence]\n"
        "  - name: spider_dev\n"
        f"    url: {SPIDER_URL}\n"
        "    archive_member: spider_data/dev.json\n"
        "    fields: [question, query]\n",
        encoding="utf-8",
    )


@pytest.fixture()
def offline_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlpup.data import download

    payloads = {
        BIRD_URL: _zip_bytes({"dev_20240627/dev.json": BIRD_FIXTURE.read_bytes()}),
        SPIDER_URL: _zip_bytes({"spider_data/dev.json": SPIDER_FIXTURE.read_bytes()}),
    }

    def fake_download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payloads[url])

    monkeypatch.setattr(download, "_download_artifact", fake_download)


def test_build_eval_groups_maps_bird_sql_and_spider_query(
    tmp_path: Path, offline_eval: None
) -> None:
    from sqlpup.config import load_eval_sets_config

    cfg_path = tmp_path / "eval_sets.yaml"
    _write_eval_config(cfg_path)
    specs = (
        EvalGroupSpec("bird_dev_question", "bird_dev", "question"),
        EvalGroupSpec("bird_dev_sql", "bird_dev", "SQL"),
        EvalGroupSpec("spider_dev_question", "spider_dev", "question"),
        EvalGroupSpec("spider_dev_sql", "spider_dev", "query"),
    )
    groups = build_eval_groups(load_eval_sets_config(cfg_path), tmp_path / "eval", specs)

    by_name = {g.name: list(g.texts) for g in groups}
    bird = json.loads(BIRD_FIXTURE.read_text(encoding="utf-8"))
    spider = json.loads(SPIDER_FIXTURE.read_text(encoding="utf-8"))

    assert by_name["bird_dev_question"] == [e["question"] for e in bird]
    assert by_name["bird_dev_sql"] == [e["SQL"] for e in bird]  # BIRD's SQL field
    assert by_name["spider_dev_question"] == [e["question"] for e in spider]
    assert by_name["spider_dev_sql"] == [e["query"] for e in spider]  # Spider's query field
    # the SQL group must be gold SQL, never the natural-language question
    assert by_name["bird_dev_sql"] != by_name["bird_dev_question"]


# --- markdown golden --------------------------------------------------------


def test_render_markdown_matches_golden() -> None:
    corpus_tokens = {
        "sqlpup-bpe32k": 50,
        "gpt-4 (cl100k_base)": 80,
        "gpt-4o (o200k_base)": 60,
        "qwen2.5-0.5b": 100,
    }
    bird_tokens = {
        "sqlpup-bpe32k": 30,
        "gpt-4 (cl100k_base)": 50,
        "gpt-4o (o200k_base)": 25,
        "qwen2.5-0.5b": 60,
    }
    report = FertilityReport(
        generated_at="2026-07-21T00:00:00+00:00",
        corpus="data/clean/corpus_sample.jsonl.zst",
        holdout_mod=50,
        eval_dir="data/eval",
        tokenizers=(
            TokenizerInfo("sqlpup-bpe32k", 32768),
            TokenizerInfo("gpt-4 (cl100k_base)", 100277),
            TokenizerInfo("gpt-4o (o200k_base)", 200019),
            TokenizerInfo("qwen2.5-0.5b", 151665),
        ),
        corpus_groups=(GroupResult("starcoder_sql", 10, 200, 40, dict(corpus_tokens)),),
        corpus_overall=GroupResult("overall", 10, 200, 40, dict(corpus_tokens)),
        eval_groups=(GroupResult("bird_dev_sql", 5, 100, 20, dict(bird_tokens)),),
    )
    golden = (FIXTURES / "fertility_golden.md").read_text(encoding="utf-8")
    assert render_markdown(report) == golden


# --- matched-vocabulary section ---------------------------------------------

MATCHED_TOKENIZERS = (
    TokenizerInfo("sqlpup-bpe32k", 32768),
    TokenizerInfo("sqlpup-100k", 100277),
    TokenizerInfo("gpt-4 (cl100k_base)", 100277),
    TokenizerInfo("gpt-4o (o200k_base)", 200019),
    TokenizerInfo("qwen2.5-0.5b", 151665),
)


def _matched_report(
    bird: dict[str, int], spider: dict[str, int], starcoder: dict[str, int]
) -> FertilityReport:
    """A 5-tokenizer report (sqlpup-100k flagged matched) over the three SQL groups."""
    return FertilityReport(
        generated_at="2026-07-21T00:00:00+00:00",
        corpus="data/clean/corpus_sample.jsonl.zst",
        holdout_mod=50,
        eval_dir="data/eval",
        tokenizers=MATCHED_TOKENIZERS,
        corpus_groups=(GroupResult("starcoder_sql", 10, 200, 40, dict(starcoder)),),
        corpus_overall=GroupResult("overall", 10, 200, 40, dict(starcoder)),
        eval_groups=(
            GroupResult("bird_dev_sql", 5, 100, 20, dict(bird)),
            GroupResult("spider_dev_sql", 4, 80, 16, dict(spider)),
        ),
        matched=("sqlpup-100k",),
    )


def _base_tokens_per_word_header(md: str) -> str:
    """The header row of the base (non-matched) tokens-per-word table."""
    return next(line for line in md.splitlines() if line.startswith("| group | docs | words |"))


def test_render_markdown_matched_section_reports_a_win() -> None:
    # sqlpup-100k uses fewer tokens than cl100k on every SQL group.
    report = _matched_report(
        bird={
            "sqlpup-bpe32k": 30,
            "sqlpup-100k": 18,
            "gpt-4 (cl100k_base)": 20,
            "gpt-4o (o200k_base)": 19,
            "qwen2.5-0.5b": 25,
        },
        spider={
            "sqlpup-bpe32k": 28,
            "sqlpup-100k": 17,
            "gpt-4 (cl100k_base)": 20,
            "gpt-4o (o200k_base)": 18,
            "qwen2.5-0.5b": 22,
        },
        starcoder={
            "sqlpup-bpe32k": 60,
            "sqlpup-100k": 45,
            "gpt-4 (cl100k_base)": 50,
            "gpt-4o (o200k_base)": 48,
            "qwen2.5-0.5b": 55,
        },
    )
    md = render_markdown(report)

    # the matched section is added and the regen hint names the matched target
    assert "## Matched-vocabulary comparison" in md
    assert "Regenerate with `make fertility-matched`" in md

    # the base 32k table is intact: sqlpup-100k is NOT a base column
    base_header = _base_tokens_per_word_header(md)
    assert "sqlpup-100k" not in base_header
    assert base_header.count("|") == 8  # group, docs, words + 4 base tokenizers

    # the matched table carries all five tokenizers with their vocab sizes
    assert "sqlpup-100k (100,277)" in md
    assert "gpt-4 (cl100k_base) (100,277)" in md
    assert "sqlpup-bpe32k (32,768)" in md

    # honest, data-driven headline + per-group margins (win on all three)
    assert "every SQL surface measured" in md
    assert "uses 10.0% fewer tokens than GPT-4 cl100k (18 vs 20)" in md  # bird
    assert "uses 15.0% fewer tokens than GPT-4 cl100k (17 vs 20)" in md  # spider
    assert "uses 10.0% fewer tokens than GPT-4 cl100k (45 vs 50)" in md  # starcoder

    # the three mandated caveats are present
    assert "Less and narrower" in md
    assert "Experiment, not the shipped" in md
    assert "Compression, not accuracy" in md


def test_render_markdown_matched_section_reports_a_loss() -> None:
    # sqlpup-100k uses MORE tokens than cl100k on every SQL group.
    report = _matched_report(
        bird={
            "sqlpup-bpe32k": 30,
            "sqlpup-100k": 22,
            "gpt-4 (cl100k_base)": 20,
            "gpt-4o (o200k_base)": 19,
            "qwen2.5-0.5b": 25,
        },
        spider={
            "sqlpup-bpe32k": 28,
            "sqlpup-100k": 23,
            "gpt-4 (cl100k_base)": 20,
            "gpt-4o (o200k_base)": 18,
            "qwen2.5-0.5b": 22,
        },
        starcoder={
            "sqlpup-bpe32k": 60,
            "sqlpup-100k": 55,
            "gpt-4 (cl100k_base)": 50,
            "gpt-4o (o200k_base)": 48,
            "qwen2.5-0.5b": 55,
        },
    )
    md = render_markdown(report)

    assert "## Matched-vocabulary comparison" in md
    assert "does not compress SQL more tightly" in md
    assert "uses 10.0% more tokens than GPT-4 cl100k (22 vs 20)" in md  # bird
    assert "uses 15.0% more tokens than GPT-4 cl100k (23 vs 20)" in md  # spider


def test_render_markdown_without_matched_is_byte_identical_to_base() -> None:
    # An empty matched tuple must reproduce the existing base report exactly.
    report = FertilityReport(
        generated_at="2026-07-21T00:00:00+00:00",
        corpus="data/clean/corpus_sample.jsonl.zst",
        holdout_mod=50,
        eval_dir="data/eval",
        tokenizers=(
            TokenizerInfo("sqlpup-bpe32k", 32768),
            TokenizerInfo("gpt-4 (cl100k_base)", 100277),
            TokenizerInfo("gpt-4o (o200k_base)", 200019),
            TokenizerInfo("qwen2.5-0.5b", 151665),
        ),
        corpus_groups=(
            GroupResult(
                "starcoder_sql",
                10,
                200,
                40,
                {
                    "sqlpup-bpe32k": 50,
                    "gpt-4 (cl100k_base)": 80,
                    "gpt-4o (o200k_base)": 60,
                    "qwen2.5-0.5b": 100,
                },
            ),
        ),
        corpus_overall=GroupResult(
            "overall",
            10,
            200,
            40,
            {
                "sqlpup-bpe32k": 50,
                "gpt-4 (cl100k_base)": 80,
                "gpt-4o (o200k_base)": 60,
                "qwen2.5-0.5b": 100,
            },
        ),
        eval_groups=(
            GroupResult(
                "bird_dev_sql",
                5,
                100,
                20,
                {
                    "sqlpup-bpe32k": 30,
                    "gpt-4 (cl100k_base)": 50,
                    "gpt-4o (o200k_base)": 25,
                    "qwen2.5-0.5b": 60,
                },
            ),
        ),
    )
    md = render_markdown(report)
    assert "Matched-vocabulary comparison" not in md
    assert "Regenerate with `make fertility`." in md


# --- CLI end-to-end (offline: real tiny BPE + fake references) --------------

TINY_SQL_CORPUS = [
    "SELECT name, age FROM users WHERE age > 21 ORDER BY name;",
    "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);",
    "INSERT INTO users (name, age) VALUES ('ada', 36);",
] * 40


def _train_tiny_tokenizer(tmp_path: Path, out_name: str = "tok") -> Path:
    from sqlpup.tokenizer.train import train_bpe

    config = TokenizerConfig(
        vocab_size=400,
        special_tokens=("<|end|>",),
        sample_docs=1000,
        out_dir=tmp_path / out_name,
    )
    return train_bpe(config, TINY_SQL_CORPUS)


def _fake_references() -> list[fertility.TokenizerAdapter]:
    return [
        WordAdapter("gpt-4 (cl100k_base)", 100_277, k=2),
        WordAdapter("gpt-4o (o200k_base)", 200_019, k=1),
        CharAdapter("qwen2.5-0.5b", 151_665),
    ]


def test_cli_fertility_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    offline_eval: None,
) -> None:
    tokenizer_path = _train_tiny_tokenizer(tmp_path)

    # a tiny two-source corpus with enough docs that holdout-mod 2 keeps some of each
    corpus = tmp_path / "corpus_sample.jsonl.zst"
    docs = [
        Document(
            id=f"sc-{i}",
            text=f"sc-{i} SELECT a FROM t WHERE a > {i};",
            source="starcoder_sql",
        )
        for i in range(20)
    ] + [
        Document(
            id=f"fw-{i}",
            text=f"fw-{i} the quick brown fox jumps over {i} dogs.",
            source="fineweb_edu",
        )
        for i in range(20)
    ]
    write_documents(corpus, docs)

    cfg_path = tmp_path / "eval_sets.yaml"
    _write_eval_config(cfg_path)

    # fake the network-only reference adapters; the eval loaders run for real
    # against monkeypatched fixture downloads (see offline_eval fixture).
    monkeypatch.setattr(fertility, "load_reference_adapters", lambda *a, **k: _fake_references())

    out_json = tmp_path / "artifacts" / "fertility.json"
    out_md = tmp_path / "docs" / "tokenizer-fertility.md"
    from sqlpup.cli import main

    code = main(
        [
            "tokenizer",
            "fertility",
            "--tokenizer",
            str(tokenizer_path),
            "--corpus",
            str(corpus),
            "--holdout-mod",
            "2",
            "--eval-config",
            str(cfg_path),
            "--eval-dir",
            str(tmp_path / "eval"),
            "--out",
            str(out_json),
            "--report",
            str(out_md),
        ]
    )
    assert code == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["out"] == str(out_json)
    assert summary["report"] == str(out_md)
    assert summary["tokenizers"]["sqlpup-bpe32k"] > 0
    assert summary["tokenizers"]["gpt-4 (cl100k_base)"] == 100_277

    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    names = [t["name"] for t in payload["tokenizers"]]
    assert names[0] == "sqlpup-bpe32k"  # sqlpup is always the first column
    assert {"gpt-4 (cl100k_base)", "gpt-4o (o200k_base)", "qwen2.5-0.5b"}.issubset(set(names))

    groups = {g["group"]: g for g in payload["groups"]}
    # corpus source groups + overall + all four eval groups are present
    assert {"starcoder_sql", "fineweb_edu", "overall"}.issubset(groups.keys())
    assert {"bird_dev_question", "bird_dev_sql", "spider_dev_sql"}.issubset(groups.keys())

    # ratios in the artifact are internally consistent for every group/tokenizer
    for g in payload["groups"]:
        for stats in g["tokenizers"].values():
            if g["words"]:
                assert stats["tokens_per_word"] == pytest.approx(stats["tokens"] / g["words"])
            if stats["tokens"]:
                assert stats["bytes_per_token"] == pytest.approx(g["bytes"] / stats["tokens"])

    md = out_md.read_text(encoding="utf-8")
    assert md.startswith("# sqlpup tokenizer fertility")
    assert "vocab size" in md
    assert "tokens per word" in md.lower()
    assert "bytes per token" in md.lower()
    assert "min_chars=200" in md  # the post-clean-filter caveat is present


def test_cli_fertility_missing_report_extra_prints_friendly_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_tiktoken(*a: object, **k: object) -> list[fertility.TokenizerAdapter]:
        raise ModuleNotFoundError("No module named 'tiktoken'")

    monkeypatch.setattr(fertility, "load_reference_adapters", _no_tiktoken)
    from sqlpup.cli import main

    code = main(
        [
            "tokenizer",
            "fertility",
            "--tokenizer",
            "nonexistent.json",
            "--corpus",
            "nonexistent.jsonl.zst",
        ]
    )
    assert code != 0
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing leaks onto the stats stream
    assert "sqlpup[report]" in captured.err  # names the actionable extra
    assert "uv sync --extra report" in captured.err
    assert "Traceback" not in captured.err  # friendly message, not a stack trace


def test_cli_fertility_extra_tokenizer_adds_matched_column(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    offline_eval: None,
) -> None:
    primary = _train_tiny_tokenizer(tmp_path, out_name="tok32k")
    matched = _train_tiny_tokenizer(tmp_path, out_name="tok100k")

    corpus = tmp_path / "corpus_sample.jsonl.zst"
    docs = [
        Document(
            id=f"sc-{i}", text=f"sc-{i} SELECT a FROM t WHERE a > {i};", source="starcoder_sql"
        )
        for i in range(20)
    ] + [
        Document(
            id=f"fw-{i}",
            text=f"fw-{i} the quick brown fox jumps over {i} dogs.",
            source="fineweb_edu",
        )
        for i in range(20)
    ]
    write_documents(corpus, docs)

    cfg_path = tmp_path / "eval_sets.yaml"
    _write_eval_config(cfg_path)
    monkeypatch.setattr(fertility, "load_reference_adapters", lambda *a, **k: _fake_references())

    out_json = tmp_path / "artifacts" / "fertility.json"
    out_md = tmp_path / "docs" / "tokenizer-fertility.md"
    from sqlpup.cli import main

    code = main(
        [
            "tokenizer",
            "fertility",
            "--tokenizer",
            str(primary),
            "--extra-tokenizer",
            f"{matched}:sqlpup-100k",
            "--corpus",
            str(corpus),
            "--holdout-mod",
            "2",
            "--eval-config",
            str(cfg_path),
            "--eval-dir",
            str(tmp_path / "eval"),
            "--out",
            str(out_json),
            "--report",
            str(out_md),
        ]
    )
    assert code == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["matched"] == ["sqlpup-100k"]
    assert summary["tokenizers"]["sqlpup-100k"] > 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    names = [t["name"] for t in payload["tokenizers"]]
    assert names[0] == "sqlpup-bpe32k"  # primary sqlpup stays the first column
    assert names[1] == "sqlpup-100k"  # the extra sqlpup adapter is second
    assert {"gpt-4 (cl100k_base)", "gpt-4o (o200k_base)", "qwen2.5-0.5b"}.issubset(set(names))
    assert payload["matched"] == ["sqlpup-100k"]

    md = out_md.read_text(encoding="utf-8")
    assert "## Matched-vocabulary comparison" in md
    assert "sqlpup-100k" in md


def test_cli_fertility_extra_tokenizer_requires_path_and_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fertility, "load_reference_adapters", lambda *a, **k: _fake_references())
    from sqlpup.cli import main

    with pytest.raises(SystemExit):
        main(
            [
                "tokenizer",
                "fertility",
                "--tokenizer",
                "primary.json",
                "--extra-tokenizer",
                "missing-label.json",  # no PATH:LABEL colon
                "--corpus",
                "corpus.jsonl.zst",
            ]
        )


def test_cli_fertility_extra_tokenizer_rejects_reserved_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A label equal to the primary sqlpup column or any reference display name
    # would silently drop a real column (or collide its counts), so it must be
    # rejected before any work. Guarded fully offline.
    monkeypatch.setattr(fertility, "load_reference_adapters", lambda *a, **k: _fake_references())
    from sqlpup.cli import main

    for reserved in (
        fertility.SQLPUP_NAME,
        fertility.GPT4_NAME,
        fertility.GPT4O_NAME,
        fertility.QWEN_NAME,
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "tokenizer",
                    "fertility",
                    "--tokenizer",
                    "primary.json",
                    "--extra-tokenizer",
                    f"some/path/tokenizer.json:{reserved}",
                    "--corpus",
                    "corpus.jsonl.zst",
                ]
            )
        assert reserved in str(excinfo.value)
