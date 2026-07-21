"""Text normalization and quality filtering.

Normalization is intentionally conservative: NFC unicode form, unified line
endings, stripped null bytes, and collapsed blank-line runs. Filtering drops
documents that are too short to teach anything, absurdly long (single-file
dumps), or dominated by non-printable characters (binary misdetected as text).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from sqlpup.io import Document

_BLANK_RUN = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class CleanConfig:
    min_chars: int = 200
    max_chars: int = 1_000_000
    max_nonprintable_ratio: float = 0.05


@dataclass(slots=True)
class CleanStats:
    kept: int = 0
    dropped_short: int = 0
    dropped_long: int = 0
    dropped_nonprintable: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "kept": self.kept,
            "dropped_short": self.dropped_short,
            "dropped_long": self.dropped_long,
            "dropped_nonprintable": self.dropped_nonprintable,
        }


def normalize_text(text: str) -> str:
    """NFC-normalize, unify newlines, strip null bytes, collapse blank runs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = unicodedata.normalize("NFC", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def nonprintable_ratio(text: str) -> float:
    """Fraction of control/format characters, excluding newlines and tabs."""
    if not text:
        return 1.0
    bad = sum(1 for ch in text if ch not in "\n\t" and unicodedata.category(ch) in {"Cc", "Cf"})
    return bad / len(text)


class Cleaner:
    """Streaming cleaner; inspect :attr:`stats` after the iterator is consumed."""

    def __init__(self, config: CleanConfig | None = None) -> None:
        self.config = config or CleanConfig()
        self.stats = CleanStats()

    def run(self, docs: Iterable[Document]) -> Iterator[Document]:
        cfg = self.config
        for doc in docs:
            text = normalize_text(doc.text)
            if len(text) < cfg.min_chars:
                self.stats.dropped_short += 1
                continue
            if len(text) > cfg.max_chars:
                self.stats.dropped_long += 1
                continue
            if nonprintable_ratio(text) > cfg.max_nonprintable_ratio:
                self.stats.dropped_nonprintable += 1
                continue
            self.stats.kept += 1
            yield Document(id=doc.id, text=text, source=doc.source, meta=doc.meta)
