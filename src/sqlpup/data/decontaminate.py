"""N-gram decontamination against evaluation sets.

Any training document sharing at least ``threshold`` distinct word-level
13-grams with an evaluation text is dropped. Shingles are lowercased and
punctuation-stripped before hashing so cosmetic differences don't hide
contamination. The index is persisted to JSON so the decontamination run
that produced a corpus is exactly reproducible and publishable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xxhash

from sqlpup.io import Document

_TOKEN = re.compile(r"[a-z0-9]+")

DEFAULT_NGRAM = 13


def word_shingles(text: str, n: int) -> set[int]:
    """Hashed word-level n-grams of ``text``; empty when fewer than n words."""
    words = _TOKEN.findall(text.lower())
    if len(words) < n:
        return set()
    return {
        int(xxhash.xxh64_intdigest(" ".join(words[i : i + n]).encode("utf-8")))
        for i in range(len(words) - n + 1)
    }


@dataclass(frozen=True, slots=True)
class NgramIndex:
    n: int
    grams: frozenset[int]

    @classmethod
    def build(cls, texts: Iterable[str], n: int = DEFAULT_NGRAM) -> NgramIndex:
        grams: set[int] = set()
        for text in texts:
            grams |= word_shingles(text, n)
        return cls(n=n, grams=frozenset(grams))

    def overlap(self, text: str) -> int:
        return len(word_shingles(text, self.n) & self.grams)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"n": self.n, "grams": sorted(self.grams)}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> NgramIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(n=int(payload["n"]), grams=frozenset(int(g) for g in payload["grams"]))


@dataclass(slots=True)
class DecontamStats:
    kept: int = 0
    dropped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"kept": self.kept, "dropped": self.dropped}


class Decontaminator:
    """Streaming filter dropping documents that overlap the eval-set index."""

    def __init__(self, index: NgramIndex, threshold: int = 1) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.index = index
        self.threshold = threshold
        self.stats = DecontamStats()

    def filter(self, docs: Iterable[Document]) -> Iterator[Document]:
        for doc in docs:
            if self.index.overlap(doc.text) >= self.threshold:
                self.stats.dropped += 1
                continue
            self.stats.kept += 1
            yield doc


def load_eval_texts(paths: Iterable[Path]) -> list[str]:
    """Extract "question SQL" strings from BIRD/Spider-style JSON files.

    Both benchmarks ship lists of objects carrying a natural-language
    ``question`` and a gold query under ``SQL``/``query``/``sql`` depending on
    the release.
    """
    texts: list[str] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list of examples")
        for item in data:
            if not isinstance(item, Mapping):
                raise ValueError(f"{path}: expected objects in the list")
            entry: Mapping[str, Any] = item
            question = str(entry.get("question", ""))
            sql = str(entry.get("SQL") or entry.get("query") or entry.get("sql") or "")
            combined = f"{question} {sql}".strip()
            if combined:
                texts.append(combined)
    return texts
