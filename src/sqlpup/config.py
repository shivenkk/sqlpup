"""Typed configuration loading for pipeline stages.

Configs are plain YAML files parsed into frozen dataclasses with strict key
validation: unknown keys and missing required keys both raise
:class:`ConfigError` with the offending file in the message, so a typo in a
config fails loudly at startup instead of silently mis-running a stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is malformed or fails validation."""


# How a source's raw records are fetched: "hf" streams from the Hugging Face
# Hub; "schemapile" downloads and parses the SchemaPile gzipped-JSON artifact;
# "synsql" streams the SynSQL data.json array and joins schema DDL from a
# cached tables.json by db_id.
SOURCE_KINDS = frozenset({"hf", "schemapile", "synsql"})

# Pre-tokenizer strategy for the byte-level BPE tokenizer. "byte_level" is the
# vanilla GPT-2 ByteLevel split (the default; unchanged behavior). "code" adds a
# cl100k-style regex split (<=3-digit number groups + code-aware whitespace)
# before the byte-level mapping, which can compress code/SQL more tightly. Both
# keep byte-level as the last step, so either round-trips arbitrary text exactly.
PRE_TOKENIZERS = frozenset({"byte_level", "code"})


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """One corpus source in the pretraining mix."""

    name: str
    hf_path: str | None
    hf_name: str | None
    split: str
    license: str
    target_tokens: int
    text_field: str | None
    doc_cap: int | None = None
    kind: str = "hf"
    url: str | None = None
    tables_url: str | None = None
    # Directory-style HF subsets (e.g. starcoderdata/sql, the-stack-dedup's
    # data/python) are folders in the repo selected via ``data_dir=``, not the
    # builder-config ``name=``. Sources for those set ``hf_data_dir`` (and leave
    # ``hf_name`` null); everything else keeps using ``hf_name``.
    hf_data_dir: str | None = None


@dataclass(frozen=True, slots=True)
class MixConfig:
    """The full pretraining data mix."""

    seed: int
    out_dir: Path
    sources: tuple[SourceConfig, ...]

    def source(self, name: str) -> SourceConfig:
        for src in self.sources:
            if src.name == name:
                return src
        known = ", ".join(s.name for s in self.sources)
        raise ConfigError(f"unknown source {name!r}; configured sources: {known}")


@dataclass(frozen=True, slots=True)
class EvalSourceConfig:
    """One evaluation set decontaminated against (BIRD/Spider dev or test)."""

    name: str
    url: str
    fields: tuple[str, ...]
    archive_member: str | None = None
    expected_examples: int | None = None


@dataclass(frozen=True, slots=True)
class EvalSetsConfig:
    """The evaluation sets feeding the decontamination n-gram index."""

    sources: tuple[EvalSourceConfig, ...]

    def source(self, name: str) -> EvalSourceConfig:
        for src in self.sources:
            if src.name == name:
                return src
        known = ", ".join(s.name for s in self.sources)
        raise ConfigError(f"unknown eval source {name!r}; configured sources: {known}")


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Settings for training the byte-level BPE tokenizer."""

    vocab_size: int
    special_tokens: tuple[str, ...]
    sample_docs: int
    out_dir: Path
    pre_tokenizer: str = "byte_level"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return raw


def _check_keys(raw: dict[str, Any], required: set[str], optional: set[str], ctx: str) -> None:
    missing = required - raw.keys()
    unknown = raw.keys() - required - optional
    if missing:
        raise ConfigError(f"{ctx}: missing required keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{ctx}: unknown keys: {sorted(unknown)}")


def _source_from(raw: Any, index: int, path: Path) -> SourceConfig:
    ctx = f"{path}: sources[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{ctx}: expected a mapping")
    _check_keys(
        raw,
        required={"name", "hf_path", "hf_name", "split", "license", "target_tokens", "text_field"},
        optional={"doc_cap", "kind", "url", "tables_url", "hf_data_dir"},
        ctx=ctx,
    )
    raw_kind = raw.get("kind", "hf")
    if raw_kind is None:
        raise ConfigError(f"{ctx}: kind may not be null")
    kind = str(raw_kind)
    if kind not in SOURCE_KINDS:
        raise ConfigError(
            f"{ctx}: unknown source kind {kind!r}; expected one of {sorted(SOURCE_KINDS)}"
        )
    return SourceConfig(
        name=str(raw["name"]),
        hf_path=None if raw["hf_path"] is None else str(raw["hf_path"]),
        hf_name=None if raw["hf_name"] is None else str(raw["hf_name"]),
        split=str(raw["split"]),
        license=str(raw["license"]),
        target_tokens=int(raw["target_tokens"]),
        text_field=None if raw["text_field"] is None else str(raw["text_field"]),
        doc_cap=None if raw.get("doc_cap") is None else int(raw["doc_cap"]),
        kind=kind,
        url=None if raw.get("url") is None else str(raw["url"]),
        tables_url=None if raw.get("tables_url") is None else str(raw["tables_url"]),
        hf_data_dir=None if raw.get("hf_data_dir") is None else str(raw["hf_data_dir"]),
    )


def load_mix_config(path: Path) -> MixConfig:
    """Load and validate a pretraining mix config."""
    raw = _load_yaml(path)
    _check_keys(raw, required={"seed", "out_dir", "sources"}, optional=set(), ctx=str(path))
    if not isinstance(raw["sources"], list) or not raw["sources"]:
        raise ConfigError(f"{path}: 'sources' must be a non-empty list")
    sources = tuple(_source_from(item, i, path) for i, item in enumerate(raw["sources"]))
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ConfigError(f"{path}: duplicate source names")
    return MixConfig(seed=int(raw["seed"]), out_dir=Path(str(raw["out_dir"])), sources=sources)


def _eval_source_from(raw: Any, index: int, path: Path) -> EvalSourceConfig:
    ctx = f"{path}: sources[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{ctx}: expected a mapping")
    _check_keys(
        raw,
        required={"name", "url", "fields"},
        optional={"archive_member", "expected_examples"},
        ctx=ctx,
    )
    fields = raw["fields"]
    if not isinstance(fields, list) or not fields or not all(isinstance(f, str) for f in fields):
        raise ConfigError(f"{ctx}: 'fields' must be a non-empty list of strings")
    return EvalSourceConfig(
        name=str(raw["name"]),
        url=str(raw["url"]),
        fields=tuple(fields),
        archive_member=(None if raw.get("archive_member") is None else str(raw["archive_member"])),
        expected_examples=(
            None if raw.get("expected_examples") is None else int(raw["expected_examples"])
        ),
    )


def load_eval_sets_config(path: Path) -> EvalSetsConfig:
    """Load and validate the evaluation-set decontamination config."""
    raw = _load_yaml(path)
    _check_keys(raw, required={"sources"}, optional=set(), ctx=str(path))
    if not isinstance(raw["sources"], list) or not raw["sources"]:
        raise ConfigError(f"{path}: 'sources' must be a non-empty list")
    sources = tuple(_eval_source_from(item, i, path) for i, item in enumerate(raw["sources"]))
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ConfigError(f"{path}: duplicate source names")
    return EvalSetsConfig(sources=sources)


def load_tokenizer_config(path: Path) -> TokenizerConfig:
    """Load and validate a tokenizer training config."""
    raw = _load_yaml(path)
    _check_keys(
        raw,
        required={"vocab_size", "special_tokens", "sample_docs", "out_dir"},
        optional={"pre_tokenizer"},
        ctx=str(path),
    )
    specials = raw["special_tokens"]
    if not isinstance(specials, list) or not all(isinstance(t, str) for t in specials):
        raise ConfigError(f"{path}: 'special_tokens' must be a list of strings")
    pre_tokenizer = str(raw.get("pre_tokenizer", "byte_level"))
    if pre_tokenizer not in PRE_TOKENIZERS:
        raise ConfigError(
            f"{path}: unknown pre_tokenizer {pre_tokenizer!r}; "
            f"expected one of {sorted(PRE_TOKENIZERS)}"
        )
    return TokenizerConfig(
        vocab_size=int(raw["vocab_size"]),
        special_tokens=tuple(specials),
        sample_docs=int(raw["sample_docs"]),
        out_dir=Path(str(raw["out_dir"])),
        pre_tokenizer=pre_tokenizer,
    )
