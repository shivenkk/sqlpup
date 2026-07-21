from pathlib import Path

import pytest

from sqlpup.config import (
    ConfigError,
    load_eval_sets_config,
    load_mix_config,
    load_tokenizer_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shipped_mix_config_loads() -> None:
    cfg = load_mix_config(REPO_ROOT / "configs" / "data" / "mix_baseline.yaml")
    assert cfg.seed == 1337
    assert [s.name for s in cfg.sources] == [
        "fineweb_edu",
        "starcoder_sql",
        "synsql",
        "schemapile",
        "stack_python",
    ]
    assert cfg.source("fineweb_edu").target_tokens == 4_000_000_000
    assert cfg.source("synsql").text_field is None
    # hf sources default to kind "hf"; schemapile is a fetched HTTP artifact
    assert cfg.source("fineweb_edu").kind == "hf"
    schemapile = cfg.source("schemapile")
    assert schemapile.kind == "schemapile"
    assert schemapile.url is not None and schemapile.url.endswith("schemapile-perm.json.gz")
    # Directory-style HF subsets are selected via data_dir, not the builder-config
    # name (which stays null), while FineWeb-Edu keeps its builder-config name.
    assert cfg.source("fineweb_edu").hf_name == "sample-10BT"
    assert cfg.source("fineweb_edu").hf_data_dir is None
    assert cfg.source("starcoder_sql").hf_name is None
    assert cfg.source("starcoder_sql").hf_data_dir == "sql"
    assert cfg.source("stack_python").hf_name is None
    assert cfg.source("stack_python").hf_data_dir == "data/python"


def test_shipped_sample_mix_config_mirrors_baseline() -> None:
    sample = load_mix_config(REPO_ROOT / "configs" / "data" / "mix_sample.yaml")
    baseline = load_mix_config(REPO_ROOT / "configs" / "data" / "mix_baseline.yaml")

    # Same sources, in the same order, streamed to a distinct out_dir.
    assert [s.name for s in sample.sources] == [s.name for s in baseline.sources]
    assert sample.out_dir == Path("data/sample")
    assert baseline.out_dir != sample.out_dir

    # ~1/100 slice per the brief; proportions mirror the baseline mix.
    targets = {s.name: s.target_tokens for s in sample.sources}
    assert targets == {
        "fineweb_edu": 40_000_000,
        "starcoder_sql": 25_000_000,
        "synsql": 15_000_000,
        "schemapile": 5_000_000,
        "stack_python": 10_000_000,
    }

    # urls / licenses / fields / fetch schema copied verbatim from the baseline.
    for name in targets:
        s, b = sample.source(name), baseline.source(name)
        assert (s.hf_path, s.hf_name, s.hf_data_dir, s.split) == (
            b.hf_path,
            b.hf_name,
            b.hf_data_dir,
            b.split,
        )
        assert (s.license, s.text_field, s.kind) == (b.license, b.text_field, b.kind)
        assert (s.url, s.tables_url) == (b.url, b.tables_url)


def test_tokenizer_v2_mix_config_is_sql_heavier_but_fetch_fields_mirror_baseline() -> None:
    v2 = load_mix_config(REPO_ROOT / "configs" / "data" / "mix_tokenizer_v2.yaml")
    baseline = load_mix_config(REPO_ROOT / "configs" / "data" / "mix_baseline.yaml")

    # Same five sources as the baseline (order may differ), streamed to a
    # distinct corpus dir so it never clobbers the pretrain download.
    assert {s.name for s in v2.sources} == {s.name for s in baseline.sources}
    assert v2.out_dir == Path("data/tokenizer_corpus")
    assert baseline.out_dir != v2.out_dir

    # ~780M total proxy-tokens, weighted toward SQL/code/schema rather than the
    # baseline's web-English weighting: fineweb drops from the largest slice to
    # the smallest, SQL/code/schema sources grow.
    targets = {s.name: s.target_tokens for s in v2.sources}
    assert targets == {
        "starcoder_sql": 250_000_000,
        "synsql": 200_000_000,
        "schemapile": 80_000_000,
        "stack_python": 100_000_000,
        "fineweb_edu": 150_000_000,
    }
    assert sum(targets.values()) == 780_000_000
    # SQL + schema + code dominate; fineweb is a minority English anchor.
    assert targets["fineweb_edu"] < targets["starcoder_sql"]

    # urls / licenses / fields / fetch schema copied verbatim from the baseline;
    # only out_dir and target_tokens differ.
    for name in targets:
        s, b = v2.source(name), baseline.source(name)
        assert (s.hf_path, s.hf_name, s.hf_data_dir, s.split) == (
            b.hf_path,
            b.hf_name,
            b.hf_data_dir,
            b.split,
        )
        assert (s.license, s.text_field, s.kind) == (b.license, b.text_field, b.kind)
        assert (s.url, s.tables_url) == (b.url, b.tables_url)


def test_synsql_source_uses_streaming_kind_and_urls() -> None:
    cfg = load_mix_config(REPO_ROOT / "configs" / "data" / "mix_baseline.yaml")
    synsql = cfg.source("synsql")
    assert synsql.kind == "synsql"
    assert synsql.url is not None and synsql.url.endswith("data.json")
    assert synsql.tables_url is not None and synsql.tables_url.endswith("tables.json")


def test_hf_data_dir_is_optional_and_parsed(tmp_path: Path) -> None:
    path = tmp_path / "ok.yaml"
    path.write_text(
        "seed: 1\nout_dir: x\nsources:\n"
        "  - {name: dir, hf_path: org/ds, hf_name: null, split: train,"
        " license: MIT, target_tokens: 1, text_field: content, hf_data_dir: sql}\n"
        "  - {name: plain, hf_path: org/ds2, hf_name: cfg, split: train,"
        " license: MIT, target_tokens: 1, text_field: text}\n"
    )
    cfg = load_mix_config(path)
    assert cfg.source("dir").hf_data_dir == "sql"
    assert cfg.source("dir").hf_name is None
    # Omitted entirely -> None, so existing configs need no data_dir key.
    assert cfg.source("plain").hf_data_dir is None


def test_unknown_source_kind_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "seed: 1\nout_dir: x\nsources:\n"
        "  - {name: a, hf_path: p, hf_name: null, split: train,"
        " license: MIT, target_tokens: 1, text_field: text, kind: bogus}\n"
    )
    with pytest.raises(ConfigError, match="unknown source kind"):
        load_mix_config(path)


def test_null_source_kind_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "seed: 1\nout_dir: x\nsources:\n"
        "  - {name: a, hf_path: p, hf_name: null, split: train,"
        " license: MIT, target_tokens: 1, text_field: text, kind: null}\n"
    )
    with pytest.raises(ConfigError, match="kind may not be null"):
        load_mix_config(path)


def test_shipped_tokenizer_config_loads() -> None:
    cfg = load_tokenizer_config(REPO_ROOT / "configs" / "tokenizer" / "bpe32k.yaml")
    assert cfg.vocab_size == 32768
    assert "<|sql|>" in cfg.special_tokens
    # The shipped tokenizer uses the code-tuned pre-tokenizer (Task 14 A/B win).
    assert cfg.pre_tokenizer == "code"


def _tokenizer_config_text(pre_tokenizer: str | None) -> str:
    body = "vocab_size: 512\nspecial_tokens: ['<|end|>']\nsample_docs: 10\nout_dir: x\n"
    return body if pre_tokenizer is None else body + f"pre_tokenizer: {pre_tokenizer}\n"


def test_tokenizer_config_defaults_to_byte_level_when_omitted(tmp_path: Path) -> None:
    # Omitting the key keeps the original byte-level behavior, so pre-existing
    # tokenizer configs need no change.
    path = tmp_path / "tok.yaml"
    path.write_text(_tokenizer_config_text(None))
    assert load_tokenizer_config(path).pre_tokenizer == "byte_level"


def test_tokenizer_config_accepts_code_pre_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "tok.yaml"
    path.write_text(_tokenizer_config_text("code"))
    assert load_tokenizer_config(path).pre_tokenizer == "code"


def test_tokenizer_config_rejects_unknown_pre_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "tok.yaml"
    path.write_text(_tokenizer_config_text("bogus"))
    with pytest.raises(ConfigError, match="pre_tokenizer"):
        load_tokenizer_config(path)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\nout_dir: x\nsources: []\ntypo_key: 1\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_mix_config(path)


def test_missing_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\n")
    with pytest.raises(ConfigError, match="missing required keys"):
        load_mix_config(path)


def test_empty_sources_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\nout_dir: x\nsources: []\n")
    with pytest.raises(ConfigError, match="non-empty"):
        load_mix_config(path)


def test_duplicate_source_names_rejected(tmp_path: Path) -> None:
    src = (
        "  - {name: a, hf_path: p, hf_name: null, split: train,"
        " license: MIT, target_tokens: 1, text_field: text}\n"
    )
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\nout_dir: x\nsources:\n" + src + src)
    with pytest.raises(ConfigError, match="duplicate"):
        load_mix_config(path)


def test_unknown_source_lookup_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ok.yaml"
    path.write_text(
        "seed: 1\nout_dir: x\nsources:\n"
        "  - {name: a, hf_path: p, hf_name: null, split: train,"
        " license: MIT, target_tokens: 1, text_field: text}\n"
    )
    cfg = load_mix_config(path)
    with pytest.raises(ConfigError, match="unknown source"):
        cfg.source("nope")


def test_shipped_eval_sets_config_loads() -> None:
    cfg = load_eval_sets_config(REPO_ROOT / "configs" / "data" / "eval_sets.yaml")
    assert [s.name for s in cfg.sources] == ["bird_dev", "spider_dev", "spider_test"]

    bird = cfg.source("bird_dev")
    assert bird.url.endswith("dev.zip")
    assert bird.archive_member == "dev_20240627/dev.json"
    assert bird.fields == ("question", "SQL", "evidence")
    assert bird.expected_examples == 1534

    spider_dev = cfg.source("spider_dev")
    spider_test = cfg.source("spider_test")
    assert spider_dev.fields == ("question", "query")
    assert spider_dev.archive_member == "spider_data/dev.json"
    assert spider_test.archive_member == "spider_data/test.json"
    # dev and test are extracted from the same downloaded archive
    assert spider_dev.url == spider_test.url
    assert spider_dev.expected_examples == 1034
    assert spider_test.expected_examples == 2147


def test_eval_sets_requires_non_empty_string_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("sources:\n  - {name: a, url: 'http://x', fields: []}\n")
    with pytest.raises(ConfigError, match="fields"):
        load_eval_sets_config(path)


def test_eval_sets_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("sources:\n  - {name: a, url: 'http://x', fields: [question], oops: 1}\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_eval_sets_config(path)


def test_eval_sets_rejects_duplicate_names(tmp_path: Path) -> None:
    src = "  - {name: a, url: 'http://x', fields: [question]}\n"
    path = tmp_path / "bad.yaml"
    path.write_text("sources:\n" + src + src)
    with pytest.raises(ConfigError, match="duplicate"):
        load_eval_sets_config(path)
