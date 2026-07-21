"""Tokenizer fertility comparison: tokens/word and bytes/token by corpus group.

Fertility is a compression metric. For a group of texts and a tokenizer we
report two ratios, both derived from summed totals (never per-document
averages), so they are deterministic given the same inputs:

- **tokens/word** = total tokens / total whitespace-split words. Lower is
  better: fewer tokens to say the same thing.
- **bytes/token** = total UTF-8 bytes / total tokens. Higher is better: each
  token carries more text, so more context fits in a fixed window.

Exact per-group definitions:

- **bytes** = ``sum(len(text.encode("utf-8")))`` over the group's texts.
- **words** = ``sum(len(text.split()))`` (whitespace split).
- **tokens** = ``sum(adapter.encode(text))`` per tokenizer, with no
  special/BOS/EOS tokens added, so the count reflects the raw text only.

sqlpup's SQL-heavy byte-level BPE (vocab 32,768) is compared against three
general-purpose tokenizers -- GPT-4 (tiktoken ``cl100k_base``), GPT-4o
(``o200k_base``) and Qwen2.5 (``Qwen/Qwen2.5-0.5B``). tiktoken is an optional
extra (``report``); the reference adapters are only built for a real run, never
imported at module load, so the core package stays importable without it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tokenizers import Tokenizer

from sqlpup.config import EvalSetsConfig
from sqlpup.data import download
from sqlpup.data.eval_sets import (
    _load_examples,
    extract_field_texts,
    fetch_archive,
    read_member,
)
from sqlpup.io import read_documents
from sqlpup.tokenizer.train import is_holdout

SQLPUP_NAME = "sqlpup-bpe32k"
GPT4_NAME = "gpt-4 (cl100k_base)"
GPT4O_NAME = "gpt-4o (o200k_base)"
QWEN_NAME = "qwen2.5-0.5b"
QWEN_REPO = "Qwen/Qwen2.5-0.5B"

REFERENCE_CACHE_DIR = Path("data/reference_tokenizers")

REPORT_EXTRA_HINT = (
    "sqlpup tokenizer fertility requires tiktoken, the optional 'report' extra,\n"
    "which is not installed. Install it with one of:\n"
    '    pip install "sqlpup[report]"\n'
    "    uv sync --extra report"
)


# --- tokenizer adapters -----------------------------------------------------


class TokenizerAdapter(Protocol):
    """A tokenizer the metric core can measure: a name, a vocab size, a counter."""

    name: str
    vocab_size: int

    def encode(self, text: str) -> int:
        """Return the number of tokens ``text`` encodes to (no special tokens)."""
        ...


class TokenizersAdapter:
    """Adapter over a ``tokenizers.Tokenizer`` (sqlpup bpe32k or an HF tokenizer.json)."""

    def __init__(self, name: str, tokenizer: Tokenizer) -> None:
        self.name = name
        self.vocab_size: int = tokenizer.get_vocab_size()
        self._tokenizer = tokenizer

    def encode(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class TiktokenAdapter:
    """Adapter over a tiktoken ``Encoding`` (deferred import; ``report`` extra)."""

    def __init__(self, name: str, encoding: Any) -> None:
        self.name = name
        self.vocab_size: int = encoding.n_vocab
        self._encoding = encoding

    def encode(self, text: str) -> int:
        # encode_ordinary never errors on special-token-looking substrings and
        # counts them as ordinary text, which is what fertility should measure.
        return len(self._encoding.encode_ordinary(text))


def load_sqlpup_adapter(path: Path, name: str = SQLPUP_NAME) -> TokenizersAdapter:
    """Load the shipped sqlpup tokenizer.json (byte-level BPE) as an adapter."""
    return TokenizersAdapter(name, Tokenizer.from_file(str(path)))


def hf_resolve_url(repo: str, filename: str = "tokenizer.json", revision: str = "main") -> str:
    """The Hugging Face ``resolve`` URL for one file of a model repo."""
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def load_hf_tokenizer_adapter(
    name: str,
    repo: str,
    cache_dir: Path = REFERENCE_CACHE_DIR,
    filename: str = "tokenizer.json",
    revision: str = "main",
) -> TokenizersAdapter:
    """Download ``repo``'s tokenizer.json once (cached) and load it as an adapter.

    Reuses the repository's cached-HTTP helper (:func:`download._download_artifact`)
    and loads with the ``tokenizers`` library -- no ``transformers`` dependency.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{repo.replace('/', '__')}.{filename}"
    if not cache.exists():
        download._download_artifact(hf_resolve_url(repo, filename, revision), cache)
    return TokenizersAdapter(name, Tokenizer.from_file(str(cache)))


def load_tiktoken_adapter(name: str, encoding_name: str) -> TiktokenAdapter:
    """Load a tiktoken encoding as an adapter (deferred import; ``report`` extra)."""
    try:
        import tiktoken
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via the CLI path
        raise ModuleNotFoundError(REPORT_EXTRA_HINT) from exc
    return TiktokenAdapter(name, tiktoken.get_encoding(encoding_name))


def load_reference_adapters(cache_dir: Path = REFERENCE_CACHE_DIR) -> list[TokenizerAdapter]:
    """Build the three reference adapters (GPT-4, GPT-4o, Qwen2.5).

    Downloads reference tokenizer data on first use, so it is never called from
    the offline unit tests -- those inject fakes in its place.
    """
    adapters: list[TokenizerAdapter] = [
        load_tiktoken_adapter(GPT4_NAME, "cl100k_base"),
        load_tiktoken_adapter(GPT4O_NAME, "o200k_base"),
        load_hf_tokenizer_adapter(QWEN_NAME, QWEN_REPO, cache_dir),
    ]
    return adapters


# --- metric core ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextGroup:
    """A named collection of texts to measure as one fertility group."""

    name: str
    texts: Sequence[str]


@dataclass(frozen=True, slots=True)
class GroupResult:
    """Summed byte/word/token totals for one group across the measured tokenizers."""

    name: str
    docs: int
    n_bytes: int
    words: int
    tokens: dict[str, int]

    def tokens_per_word(self, tokenizer: str) -> float:
        if self.words == 0:
            return 0.0
        return self.tokens.get(tokenizer, 0) / self.words

    def bytes_per_token(self, tokenizer: str) -> float:
        count = self.tokens.get(tokenizer, 0)
        if count == 0:
            return 0.0
        return self.n_bytes / count


def measure_group(group: TextGroup, adapters: Sequence[TokenizerAdapter]) -> GroupResult:
    """Sum bytes, words and per-tokenizer token counts over ``group``'s texts.

    Token columns follow ``adapters`` order, which keeps the report's column
    order stable and deterministic.
    """
    tokens = {adapter.name: 0 for adapter in adapters}
    n_bytes = 0
    words = 0
    docs = 0
    for text in group.texts:
        docs += 1
        n_bytes += len(text.encode("utf-8"))
        words += len(text.split())
        for adapter in adapters:
            tokens[adapter.name] += adapter.encode(text)
    return GroupResult(name=group.name, docs=docs, n_bytes=n_bytes, words=words, tokens=tokens)


def aggregate_results(results: Sequence[GroupResult], name: str) -> GroupResult:
    """Combine group results into one by summing raw totals (not averaging ratios).

    Ratios must be recomputed from the summed totals, so a long document does not
    count the same as a short one; this re-sums the underlying counts instead.
    """
    tokens: dict[str, int] = {}
    docs = 0
    n_bytes = 0
    words = 0
    for result in results:
        docs += result.docs
        n_bytes += result.n_bytes
        words += result.words
        for key, value in result.tokens.items():
            tokens[key] = tokens.get(key, 0) + value
    return GroupResult(name=name, docs=docs, n_bytes=n_bytes, words=words, tokens=tokens)


# --- corpus selectors -------------------------------------------------------


def holdout_groups(corpus_path: Path, holdout_mod: int) -> list[TextGroup]:
    """Group the corpus's holdout slice by ``Document.source``.

    Selects exactly the documents the BPE trainer never saw -- those where
    :func:`sqlpup.tokenizer.train.is_holdout` is true for ``holdout_mod`` (the
    same ``xxh64(id) % N == 0`` rule the BPE trainer held out with) -- so the fertility
    numbers are measured on held-out text. Groups are returned sorted by source
    name for a deterministic row/column order.
    """
    by_source: dict[str, list[str]] = {}
    for doc in read_documents(corpus_path):
        if is_holdout(doc.id, holdout_mod):
            by_source.setdefault(doc.source, []).append(doc.text)
    return [TextGroup(source, by_source[source]) for source in sorted(by_source)]


@dataclass(frozen=True, slots=True)
class EvalGroupSpec:
    """One eval-set fertility group: a group label, its eval source, and its field."""

    group: str
    source: str
    field: str


# BIRD's gold SQL lives under "SQL"; Spider's under "query". Questions are the
# natural-language side; SQL/query are the downstream target text no tokenizer
# trained on -- the most meaningful comparison surface.
EVAL_GROUP_SPECS: tuple[EvalGroupSpec, ...] = (
    EvalGroupSpec("bird_dev_question", "bird_dev", "question"),
    EvalGroupSpec("bird_dev_sql", "bird_dev", "SQL"),
    EvalGroupSpec("spider_dev_question", "spider_dev", "question"),
    EvalGroupSpec("spider_dev_sql", "spider_dev", "query"),
)


def build_eval_groups(
    config: EvalSetsConfig, eval_dir: Path, specs: Sequence[EvalGroupSpec]
) -> list[TextGroup]:
    """Build eval-set fertility groups from the cached ``data/eval`` archives.

    Reuses the existing eval-set loaders (:func:`fetch_archive`,
    :func:`read_member`, :func:`extract_field_texts`); each source's examples are
    parsed once and reused across the group specs that reference it.
    """
    examples: dict[str, list[Mapping[str, Any]]] = {}
    groups: list[TextGroup] = []
    for spec in specs:
        if spec.source not in examples:
            source = config.source(spec.source)
            archive = fetch_archive(source, eval_dir)
            examples[spec.source] = _load_examples(
                read_member(archive, source.archive_member), source.name
            )
        texts = extract_field_texts(examples[spec.source], (spec.field,))
        groups.append(TextGroup(spec.group, texts))
    return groups


# --- report -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenizerInfo:
    """A tokenizer's display name and vocab size, shown side by side in the report."""

    name: str
    vocab_size: int


@dataclass(frozen=True, slots=True)
class FertilityReport:
    """The full fertility comparison: tokenizers, corpus groups + overall, eval groups."""

    generated_at: str
    corpus: str
    holdout_mod: int
    eval_dir: str
    tokenizers: tuple[TokenizerInfo, ...]
    corpus_groups: tuple[GroupResult, ...]
    corpus_overall: GroupResult
    eval_groups: tuple[GroupResult, ...]
    # Names of the sqlpup-side tokenizers added purely for the matched-vocabulary
    # comparison (e.g. ("sqlpup-100k",)); empty for the base 32k report. They are
    # excluded from the base tables and shown only in the matched-vocab section.
    matched: tuple[str, ...] = ()

    def _group_dict(self, result: GroupResult, kind: str) -> dict[str, Any]:
        return {
            "group": result.name,
            "kind": kind,
            "docs": result.docs,
            "bytes": result.n_bytes,
            "words": result.words,
            "tokenizers": {
                info.name: {
                    "tokens": result.tokens.get(info.name, 0),
                    "tokens_per_word": result.tokens_per_word(info.name),
                    "bytes_per_token": result.bytes_per_token(info.name),
                }
                for info in self.tokenizers
            },
        }

    def to_json_dict(self) -> dict[str, Any]:
        """The full structured artifact (written to ``--out``, gitignored)."""
        groups = [self._group_dict(r, "corpus") for r in self.corpus_groups]
        groups.append(self._group_dict(self.corpus_overall, "corpus_overall"))
        groups.extend(self._group_dict(r, "eval") for r in self.eval_groups)
        return {
            "generated_at": self.generated_at,
            "corpus": self.corpus,
            "holdout_mod": self.holdout_mod,
            "eval_dir": self.eval_dir,
            "tokenizers": [{"name": t.name, "vocab_size": t.vocab_size} for t in self.tokenizers],
            "matched": list(self.matched),
            "groups": groups,
        }

    def emit_summary(self, *, out: str, report: str | None) -> dict[str, object]:
        """A compact one-line stdout summary (the ``_emit`` convention)."""
        all_groups = [*self.corpus_groups, self.corpus_overall, *self.eval_groups]
        return {
            "out": out,
            "report": report,
            "holdout_mod": self.holdout_mod,
            "matched": list(self.matched),
            "tokenizers": {t.name: t.vocab_size for t in self.tokenizers},
            "groups": {
                group.name: {
                    t.name: {
                        "tokens_per_word": round(group.tokens_per_word(t.name), 4),
                        "bytes_per_token": round(group.bytes_per_token(t.name), 4),
                    }
                    for t in self.tokenizers
                }
                for group in all_groups
            },
        }


def build_report(
    adapters: Sequence[TokenizerAdapter],
    corpus_groups: Sequence[TextGroup],
    eval_groups: Sequence[TextGroup],
    *,
    corpus: str,
    holdout_mod: int,
    eval_dir: str,
    generated_at: str,
    matched: Sequence[str] = (),
) -> FertilityReport:
    """Measure every group with every adapter and assemble the report.

    ``matched`` names the sqlpup-side adapters added only for the
    matched-vocabulary comparison; they are measured like any other column but
    rendered in a dedicated section, never mixed into the base 32k tables.
    """
    corpus_results = [measure_group(group, adapters) for group in corpus_groups]
    overall = aggregate_results(corpus_results, name="overall")
    eval_results = [measure_group(group, adapters) for group in eval_groups]
    return FertilityReport(
        generated_at=generated_at,
        corpus=corpus,
        holdout_mod=holdout_mod,
        eval_dir=eval_dir,
        tokenizers=tuple(TokenizerInfo(a.name, a.vocab_size) for a in adapters),
        corpus_groups=tuple(corpus_results),
        corpus_overall=overall,
        eval_groups=tuple(eval_results),
        matched=tuple(matched),
    )


# --- markdown rendering -----------------------------------------------------


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_ratio(value: float) -> str:
    return f"{value:.3f}"


def _table(
    report: FertilityReport,
    tokenizers: Sequence[TokenizerInfo],
    *,
    count_header: str,
    count_of: str,
    ratio: str,
) -> list[str]:
    """Render one metric table: a count column plus one ratio column per tokenizer."""
    names = [t.name for t in tokenizers]
    header = f"| group | docs | {count_header} | " + " | ".join(names) + " |"
    divider = "| --- | ---: | ---: | " + " | ".join("---:" for _ in names) + " |"
    lines = [header, divider]

    def row(result: GroupResult, *, bold: bool) -> str:
        label = f"**{result.name}**" if bold else result.name
        count = getattr(result, count_of)
        cells = [label, _fmt_int(result.docs), _fmt_int(count)]
        cells += [_fmt_ratio(getattr(result, ratio)(t.name)) for t in tokenizers]
        return "| " + " | ".join(cells) + " |"

    lines.extend(row(r, bold=False) for r in report.corpus_groups)
    lines.append(row(report.corpus_overall, bold=True))
    lines.extend(row(r, bold=False) for r in report.eval_groups)
    return lines


def render_markdown(report: FertilityReport) -> str:
    """Render the full, standalone, regenerable markdown report."""
    matched = set(report.matched)
    # The matched-vocabulary tokenizers get their own section; the base tables
    # keep exactly the shipped reference set, so the 32k report is intact.
    base_tokenizers = tuple(t for t in report.tokenizers if t.name not in matched)
    sqlpup_vocab = base_tokenizers[0].vocab_size if base_tokenizers else 0
    regen = "make fertility-matched" if report.matched else "make fertility"
    lines: list[str] = [
        "# sqlpup tokenizer fertility",
        "",
        f"_Generated {report.generated_at}. Regenerate with `{regen}`._",
        "",
        "sqlpup ships a byte-level BPE tokenizer with a code-tuned pre-tokenizer, trained "
        "on the SQL-heavy corpus `configs/data/mix_tokenizer_v2.yaml`. It is compared "
        "against three general-purpose tokenizers with 3-6x larger vocabularies (GPT-4, "
        "GPT-4o, Qwen2.5) on held-out corpus text and on the BIRD/Spider eval sets. Lower "
        "**tokens/word** and higher **bytes/token** mean tighter compression: more text "
        "per token, so more context per training and inference step. The larger "
        "vocabularies hold more merges, so every ratio should be read against the vocab "
        "sizes below.",
        "",
        "**Before -> after.** The initial tokenizer (a small ~95M-token, ~35%-SQL "
        "balanced sample; plain byte-level pre-tokenizer) trailed GPT-4's cl100k on SQL "
        "even at matched vocabulary (BIRD dev SQL +8.0%, Spider dev SQL +9.0% more "
        "tokens). Retraining on the bigger, SQL-heavier corpus alone barely moved the "
        "out-of-distribution SQL gap (~+7.8% / +8.2%) -- it was not corpus-size-bound. "
        "Adding the code-tuned pre-tokenizer (cl100k-style <=3-digit number groups + "
        "code-aware whitespace) closed and reversed it: at matched vocabulary sqlpup now "
        "uses fewer tokens than cl100k on every SQL surface below, at a modest ~0.4-0.6% "
        "cost on English questions.",
        "",
        "## Tokenizers",
        "",
        "| tokenizer | vocab size |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {t.name} | {_fmt_int(t.vocab_size)} |" for t in base_tokenizers)
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- **Corpus provenance.** The SQL-heavy tokenizer mix "
            "(`configs/data/mix_tokenizer_v2.yaml`, ~780M proxy-tokens across five "
            f"sources) cleaned and merged into one corpus (`{report.corpus}`). Fertility "
            "is measured only on the **holdout slice** -- documents where "
            f"`xxh64(id) % {report.holdout_mod} == 0`, the text the BPE trainer never saw "
            f"(trained with `--holdout-mod {report.holdout_mod}`).",
            "- **Eval sets.** BIRD dev and Spider dev questions and gold SQL, loaded from "
            f"the cached `{report.eval_dir}` archives. No tokenizer here trained on them, "
            "so they are the most meaningful out-of-distribution comparison surface.",
            "- **Metrics.** Per group: **bytes** = UTF-8 byte length summed; **words** = "
            "whitespace split (`len(text.split())`) summed; **tokens** = ids per tokenizer "
            "summed, no special tokens added. **tokens/word** = tokens / words; "
            "**bytes/token** = bytes / tokens. Ratios come from summed totals (not "
            "per-document averages) and are deterministic.",
            "",
            "## Tokens per word (lower is better)",
            "",
        ]
    )
    lines.extend(
        _table(
            report, base_tokenizers, count_header="words", count_of="words", ratio="tokens_per_word"
        )
    )
    lines.extend(
        [
            "",
            "## Bytes per token (higher is better)",
            "",
        ]
    )
    lines.extend(
        _table(
            report,
            base_tokenizers,
            count_header="bytes",
            count_of="n_bytes",
            ratio="bytes_per_token",
        )
    )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- **Vocab-size asymmetry (read the ratios with this in mind).** sqlpup's "
            f"vocab is {_fmt_int(sqlpup_vocab)} against roughly 100k (GPT-4), 200k (GPT-4o) "
            "and 152k (Qwen2.5) -- 3-6x larger. Those larger vocabularies hold more merges "
            "and generally compress SQL more tightly here in absolute bytes/token; sqlpup's "
            "aim is competitive compression at a fraction of the embedding parameters, "
            "purpose-built for the SQL-heavy mix, not to beat far larger vocabularies on raw "
            "ratio. The asymmetry is stated, not hidden, because the ratios cannot be read "
            "fairly without it.",
            "- **Fertility is a compression metric, not task quality.** Fewer tokens per "
            "word is a throughput and context-length win; it is not by itself a measure of "
            "downstream text-to-SQL accuracy.",
            "- **The corpus holdout is same-distribution as the training mix.** It is text "
            "the trainer never saw, but drawn from the same five sources. The eval-set "
            "groups (BIRD/Spider) are the genuinely out-of-distribution surface and the "
            "more meaningful comparison.",
            "- **The corpus is post-clean-filter.** The `min_chars=200` clean filter drops "
            "short documents (here ~25k of ~400k, mostly short StarCoder SQL files and "
            "SchemaPile schemas), so very short SQL snippets are underrepresented in the "
            "corpus-slice groups -- another reason the eval-set groups matter.",
        ]
    )
    if report.matched:
        lines.extend(_matched_section(report))
    return "\n".join(lines) + "\n"


# --- matched-vocabulary section ---------------------------------------------

# The SQL surfaces compared at matched vocab, narrated in this order (the two
# out-of-distribution eval sets first, then the corpus holdout).
_MATCHED_SQL_GROUPS: tuple[tuple[str, str], ...] = (
    ("bird_dev_sql", "BIRD dev SQL"),
    ("spider_dev_sql", "Spider dev SQL"),
    ("starcoder_sql", "StarCoder SQL (corpus holdout)"),
)

# The English-question surfaces, shown alongside the SQL ones so the (small)
# natural-language cost of a SQL-tuned vocabulary is measured, not just asserted.
_MATCHED_QUESTION_GROUPS: tuple[tuple[str, str], ...] = (
    ("bird_dev_question", "BIRD dev question"),
    ("spider_dev_question", "Spider dev question"),
)

# Row order for the matched-vocab tables: corpus SQL + overall, then the eval
# question/SQL pairs (question then its gold SQL, per eval set).
_MATCHED_TABLE_GROUPS: tuple[str, ...] = (
    "starcoder_sql",
    "overall",
    "bird_dev_question",
    "bird_dev_sql",
    "spider_dev_question",
    "spider_dev_sql",
)


def _matched_table(
    results: Sequence[GroupResult],
    tokenizers: Sequence[TokenizerInfo],
    *,
    count_header: str,
    count_of: str,
    ratio: str,
) -> list[str]:
    """Render one matched-vocab metric table (vocab sizes in the column headers)."""
    heads = [f"{t.name} ({_fmt_int(t.vocab_size)})" for t in tokenizers]
    header = f"| group | docs | {count_header} | " + " | ".join(heads) + " |"
    divider = "| --- | ---: | ---: | " + " | ".join("---:" for _ in tokenizers) + " |"
    lines = [header, divider]
    for result in results:
        label = f"**{result.name}**" if result.name == "overall" else result.name
        count = getattr(result, count_of)
        cells = [label, _fmt_int(result.docs), _fmt_int(count)]
        cells += [_fmt_ratio(getattr(result, ratio)(t.name)) for t in tokenizers]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _group_margins(
    by_name: Mapping[str, GroupResult],
    matched_name: str,
    specs: Sequence[tuple[str, str]],
) -> list[tuple[str, float, int, int]]:
    """Per-group token margin (percent) of the matched sqlpup tokenizer vs GPT-4 cl100k."""
    margins: list[tuple[str, float, int, int]] = []
    for group, pretty in specs:
        result = by_name.get(group)
        if result is None:
            continue
        matched_tokens = result.tokens.get(matched_name, 0)
        cl_tokens = result.tokens.get(GPT4_NAME, 0)
        if cl_tokens == 0:
            continue
        pct = (matched_tokens - cl_tokens) / cl_tokens * 100.0
        margins.append((pretty, pct, matched_tokens, cl_tokens))
    return margins


def _margin_bullets(margins: Sequence[tuple[str, float, int, int]], matched_name: str) -> list[str]:
    """One '- **group:** uses X% fewer/more tokens ...' bullet per measured margin."""
    lines: list[str] = []
    for pretty, pct, matched_tokens, cl_tokens in margins:
        direction = "fewer" if pct < 0 else "more"
        lines.append(
            f"- **{pretty}:** {matched_name} uses {abs(pct):.1f}% {direction} tokens than "
            f"GPT-4 cl100k ({_fmt_int(matched_tokens)} vs {_fmt_int(cl_tokens)})."
        )
    return lines


def _matched_section(report: FertilityReport) -> list[str]:
    """The added matched-vocabulary comparison: headline, margins, tables, caveats.

    Compares the first matched tokenizer (the ~100k-vocab sqlpup BPE) head-to-head
    with GPT-4 cl100k at equal vocab size on the SQL surfaces, so the numbers
    isolate domain specialization from the vocab-size gap. Written straight from
    the measured totals, so the win/loss framing follows the data either way.
    """
    by_name = {
        r.name: r for r in (*report.corpus_groups, report.corpus_overall, *report.eval_groups)
    }
    matched_name = report.matched[0]

    # Per-group margin vs GPT-4 cl100k. The SQL surfaces drive the win/loss
    # headline; the question surfaces are reported as the natural-language cost.
    margins = _group_margins(by_name, matched_name, _MATCHED_SQL_GROUPS)
    question_margins = _group_margins(by_name, matched_name, _MATCHED_QUESTION_GROUPS)

    wins = sum(1 for _, pct, _, _ in margins if pct < 0)
    total = len(margins)
    if total and wins == total:
        headline = (
            "At matched vocabulary the domain-specialization effect is real: "
            f"**{matched_name}** compresses SQL more tightly than GPT-4's cl100k on "
            "every SQL surface measured below."
        )
    elif total and wins == 0:
        headline = (
            f"Even at matched vocabulary, **{matched_name}** does not compress SQL more "
            "tightly than GPT-4's cl100k on the SQL surfaces below (cl100k trained on far "
            "more and broader data; the sample corpus is only partly SQL)."
        )
    else:
        headline = (
            f"At matched vocabulary **{matched_name}** compresses SQL more tightly than "
            f"GPT-4's cl100k on {wins} of {total} SQL surfaces below."
        )

    lines = [
        "",
        "## Matched-vocabulary comparison",
        "",
        "The 32k comparison above is confounded by vocab size: GPT-4/GPT-4o/Qwen2.5 carry "
        "3-6x more merges than sqlpup's shipped 32,768-token vocabulary. To isolate "
        "*domain specialization* from *vocab size*, a throwaway sqlpup byte-level BPE was "
        "trained at GPT-4 cl100k_base's exact vocabulary (100,277) on the same "
        f"corpus (`{report.corpus}`) and the same 1/{report.holdout_mod} holdout, then "
        f"compared head-to-head. **{matched_name}** is an experiment only -- the model "
        "still ships the 32k tokenizer.",
        "",
        headline,
        "",
    ]
    lines.extend(_margin_bullets(margins, matched_name))
    if question_margins:
        lines.extend(
            [
                "",
                "The same head-to-head on the English **question** surfaces, where a "
                "SQL-tuned vocabulary is expected to give ground:",
                "",
            ]
        )
        lines.extend(_margin_bullets(question_margins, matched_name))

    results = [by_name[g] for g in _MATCHED_TABLE_GROUPS if g in by_name]
    lines.extend(["", "### Tokens per word (lower is better)", ""])
    lines.extend(
        _matched_table(
            results,
            report.tokenizers,
            count_header="words",
            count_of="words",
            ratio="tokens_per_word",
        )
    )
    lines.extend(["", "### Bytes per token (higher is better)", ""])
    lines.extend(
        _matched_table(
            results,
            report.tokenizers,
            count_header="bytes",
            count_of="n_bytes",
            ratio="bytes_per_token",
        )
    )
    lines.extend(
        [
            "",
            "### Matched-vocab caveats",
            "",
            "- **Less and narrower training data (cuts against sqlpup).** The 100k tokenizer "
            "was trained on the same ~780M-token SQL-heavy corpus -- far less and narrower "
            "text than cl100k saw. A matched-vocab win despite that handicap is strong; a "
            "loss is partly explained by it.",
            "- **Experiment, not the shipped tokenizer.** The model ships the 32k tokenizer "
            "(`artifacts/tokenizer/tokenizer.json`); this 100k tokenizer is a science "
            "artifact only and is never used for training or inference.",
            "- **Compression, not accuracy.** Fertility measures tokens per unit text, not "
            "downstream text-to-SQL accuracy.",
        ]
    )
    return lines
