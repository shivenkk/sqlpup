"""Corpus statistics and data-mix reporting.

Scans the pipeline's ``Document`` jsonl(.zst) files for the sources named in a
:class:`~sqlpup.config.MixConfig` and reports, per source and in config order,
document count, total characters, and approximate tokens -- using the same
``chars * APPROX_TOKENS_PER_CHAR`` proxy the download budgets enforce (imported
from :mod:`sqlpup.data.download`, not redefined) so the numbers line up with the
budgets they are compared against. Each source also gets its share of the actual
mix and its actual-vs-target token ratio. A missing source file is reported as
absent with zero counts rather than raising, so a partially built corpus reports
cleanly mid-build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlpup.config import MixConfig
from sqlpup.data.download import APPROX_TOKENS_PER_CHAR
from sqlpup.io import read_documents

# The download stage writes ``<name>.jsonl.zst``; a plain ``.jsonl`` is also
# accepted so uncompressed intermediate corpora can be inspected.
_CANDIDATE_SUFFIXES = (".jsonl.zst", ".jsonl")


def approx_tokens(chars: int) -> int:
    """Approximate token count from a character count (pre-tokenizer proxy)."""
    return int(chars * APPROX_TOKENS_PER_CHAR)


def find_corpus_file(corpus_dir: Path, name: str) -> Path | None:
    """First existing corpus file for ``name`` under ``corpus_dir``, else None."""
    for suffix in _CANDIDATE_SUFFIXES:
        candidate = corpus_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class SourceStats:
    name: str
    present: bool
    docs: int
    chars: int
    approx_tokens: int
    target_tokens: int
    share_pct: float
    actual_vs_target: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "present": self.present,
            "docs": self.docs,
            "chars": self.chars,
            "approx_tokens": self.approx_tokens,
            "target_tokens": self.target_tokens,
            "share_pct": self.share_pct,
            "actual_vs_target": self.actual_vs_target,
        }


@dataclass(frozen=True, slots=True)
class CorpusStats:
    sources: tuple[SourceStats, ...]
    total_docs: int
    total_chars: int
    total_approx_tokens: int
    total_target_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": [s.as_dict() for s in self.sources],
            "totals": {
                "docs": self.total_docs,
                "chars": self.total_chars,
                "approx_tokens": self.total_approx_tokens,
                "target_tokens": self.total_target_tokens,
            },
        }


def _scan_source(corpus_dir: Path, name: str) -> tuple[bool, int, int]:
    """Return (present, docs, chars) for one source, streaming its file."""
    path = find_corpus_file(corpus_dir, name)
    if path is None:
        return False, 0, 0
    docs = 0
    chars = 0
    for doc in read_documents(path):
        docs += 1
        chars += len(doc.text)
    return True, docs, chars


def compute_corpus_stats(mix: MixConfig, corpus_dir: Path) -> CorpusStats:
    """Scan ``corpus_dir`` for every source in ``mix`` (config order preserved)."""
    # per source: (name, target_tokens, present, docs, chars, approx_tokens)
    scanned: list[tuple[str, int, bool, int, int, int]] = []
    total_docs = total_chars = total_tokens = total_target = 0
    for src in mix.sources:
        present, docs, chars = _scan_source(corpus_dir, src.name)
        tokens = approx_tokens(chars)
        total_docs += docs
        total_chars += chars
        total_tokens += tokens
        total_target += src.target_tokens
        scanned.append((src.name, src.target_tokens, present, docs, chars, tokens))

    sources = tuple(
        SourceStats(
            name=name,
            present=present,
            docs=docs,
            chars=chars,
            approx_tokens=tokens,
            target_tokens=target,
            share_pct=round(100.0 * tokens / total_tokens, 4) if total_tokens else 0.0,
            actual_vs_target=round(tokens / target, 4) if target else 0.0,
        )
        for (name, target, present, docs, chars, tokens) in scanned
    )
    return CorpusStats(
        sources=sources,
        total_docs=total_docs,
        total_chars=total_chars,
        total_approx_tokens=total_tokens,
        total_target_tokens=total_target,
    )


_COLUMNS = ("source", "present", "docs", "chars", "approx_tok", "target_tok", "share%", "act/tgt")


def _row(stats: SourceStats) -> tuple[str, ...]:
    return (
        stats.name,
        "yes" if stats.present else "no",
        str(stats.docs),
        str(stats.chars),
        str(stats.approx_tokens),
        str(stats.target_tokens),
        f"{stats.share_pct:.2f}",
        f"{stats.actual_vs_target:.2f}",
    )


def render_table(stats: CorpusStats) -> str:
    """Compact aligned table; source column left-justified, numbers right."""
    total_ratio = (
        stats.total_approx_tokens / stats.total_target_tokens if stats.total_target_tokens else 0.0
    )
    total_row = (
        "TOTAL",
        "",
        str(stats.total_docs),
        str(stats.total_chars),
        str(stats.total_approx_tokens),
        str(stats.total_target_tokens),
        "100.00" if stats.total_approx_tokens else "0.00",
        f"{total_ratio:.2f}",
    )
    rows = [_COLUMNS, *(_row(s) for s in stats.sources), total_row]
    widths = [max(len(row[col]) for row in rows) for col in range(len(_COLUMNS))]
    lines = []
    for row in rows:
        cells = [
            cell.ljust(widths[col]) if col == 0 else cell.rjust(widths[col])
            for col, cell in enumerate(row)
        ]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def write_report(stats: CorpusStats, path: Path) -> dict[str, Any]:
    """Write the stats report as JSON and return the written payload."""
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "approx_tokens_per_char": APPROX_TOKENS_PER_CHAR,
        **stats.as_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
