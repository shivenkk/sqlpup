from pathlib import Path

from sqlpup.config import MixConfig, SourceConfig
from sqlpup.data.stats import CorpusStats, compute_corpus_stats, render_table
from sqlpup.io import Document, write_documents


def _src(name: str, target_tokens: int) -> SourceConfig:
    return SourceConfig(
        name=name,
        hf_path=None,
        hf_name=None,
        split="train",
        license="MIT",
        target_tokens=target_tokens,
        text_field="text",
    )


def _mix(*sources: SourceConfig) -> MixConfig:
    return MixConfig(seed=1, out_dir=Path("data/raw"), sources=tuple(sources))


def _fixture_corpus(root: Path) -> None:
    # alpha: two docs totalling 400 chars -> 100 approx tokens; written as .zst
    write_documents(
        root / "alpha.jsonl.zst",
        [
            Document(id="a1", text="x" * 100, source="alpha"),
            Document(id="a2", text="y" * 300, source="alpha"),
        ],
    )
    # beta: one doc of 800 chars -> 200 approx tokens; written as a plain .jsonl
    write_documents(root / "beta.jsonl", [Document(id="b1", text="z" * 800, source="beta")])
    # gamma: deliberately absent


def test_compute_corpus_stats_counts_shares_and_ratios(tmp_path: Path) -> None:
    _fixture_corpus(tmp_path)
    mix = _mix(_src("alpha", 200), _src("beta", 100), _src("gamma", 500))
    stats = compute_corpus_stats(mix, tmp_path)

    # deterministic config order, gamma included despite being absent
    assert [s.name for s in stats.sources] == ["alpha", "beta", "gamma"]
    by = {s.name: s for s in stats.sources}

    assert by["alpha"].present is True
    assert (by["alpha"].docs, by["alpha"].chars, by["alpha"].approx_tokens) == (2, 400, 100)
    assert by["alpha"].target_tokens == 200
    assert by["alpha"].actual_vs_target == 0.5
    assert by["alpha"].share_pct == 33.3333

    assert by["beta"].present is True
    assert (by["beta"].docs, by["beta"].chars, by["beta"].approx_tokens) == (1, 800, 200)
    assert by["beta"].actual_vs_target == 2.0
    assert by["beta"].share_pct == 66.6667

    assert by["gamma"].present is False
    assert (by["gamma"].docs, by["gamma"].chars, by["gamma"].approx_tokens) == (0, 0, 0)
    assert by["gamma"].target_tokens == 500
    assert by["gamma"].actual_vs_target == 0.0
    assert by["gamma"].share_pct == 0.0

    assert (stats.total_docs, stats.total_chars) == (3, 1200)
    assert stats.total_approx_tokens == 300
    assert stats.total_target_tokens == 800


def test_missing_corpus_dir_reports_all_absent(tmp_path: Path) -> None:
    mix = _mix(_src("alpha", 200), _src("beta", 100))
    stats = compute_corpus_stats(mix, tmp_path / "does-not-exist")
    assert all(not s.present for s in stats.sources)
    assert all(s.approx_tokens == 0 and s.share_pct == 0.0 for s in stats.sources)
    assert stats.total_approx_tokens == 0


def test_render_table_shows_per_source_and_total_lines(tmp_path: Path) -> None:
    _fixture_corpus(tmp_path)
    mix = _mix(_src("alpha", 200), _src("beta", 100), _src("gamma", 500))
    table = render_table(compute_corpus_stats(mix, tmp_path))
    lines = {ln.split()[0]: ln.split() for ln in table.splitlines() if ln.split()}

    # tolerant of column spacing: check the tokens present on each key line
    assert lines["alpha"][:6] == ["alpha", "yes", "2", "400", "100", "200"]
    assert "33.33" in lines["alpha"]
    assert "0.50" in lines["alpha"]
    assert lines["gamma"][:6] == ["gamma", "no", "0", "0", "0", "500"]
    assert lines["TOTAL"][1:5] == ["3", "1200", "300", "800"]


def test_as_dict_round_trips_structure(tmp_path: Path) -> None:
    _fixture_corpus(tmp_path)
    mix = _mix(_src("alpha", 200), _src("beta", 100))
    stats: CorpusStats = compute_corpus_stats(mix, tmp_path)
    payload = stats.as_dict()
    assert [s["name"] for s in payload["sources"]] == ["alpha", "beta"]
    assert payload["totals"] == {
        "docs": 3,
        "chars": 1200,
        "approx_tokens": 300,
        "target_tokens": 300,
    }
