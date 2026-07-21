"""Duplicate filtering: exact (content hash) and near (MinHash + LSH).

:class:`ExactDeduper` collapses byte-identical (optionally whitespace-insensitive)
documents. :class:`NearDeduper` additionally removes *near*-duplicates -- mirrored
pages, boilerplate SQL dumps, and reformatted copies that differ by a handful of
words -- via word-shingle MinHash with LSH banding. Both keep the first occurrence
and stream one document at a time; run exact first, then near on the survivors.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import xxhash

from sqlpup.data.decontaminate import word_shingles
from sqlpup.io import Document

_WS = re.compile(r"\s+")


@dataclass(slots=True)
class DedupStats:
    seen: int = 0
    unique: int = 0
    dropped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"seen": self.seen, "unique": self.unique, "dropped": self.dropped}


class ExactDeduper:
    """Streaming exact deduplicator; keeps the first occurrence of each text."""

    def __init__(self, normalize_whitespace: bool = True) -> None:
        self.normalize_whitespace = normalize_whitespace
        self.stats = DedupStats()
        self._hashes: set[int] = set()

    def _key(self, text: str) -> int:
        if self.normalize_whitespace:
            text = _WS.sub(" ", text).strip()
        return int(xxhash.xxh64_intdigest(text.encode("utf-8")))

    def filter(self, docs: Iterable[Document]) -> Iterator[Document]:
        for doc in docs:
            self.stats.seen += 1
            key = self._key(doc.text)
            if key in self._hashes:
                self.stats.dropped += 1
                continue
            self._hashes.add(key)
            self.stats.unique += 1
            yield doc


# --- near-duplicate filtering -----------------------------------------------

# MinHash permutations are realized as an overflow-safe Carter-Wegman hash
# family, h(x) = ((a * x + b) mod PRIME), over a 32-bit shingle domain. Keeping
# a, b, and the reduced shingle hashes all below 2**32 guarantees a * x + b
# stays below 2**64, so the numpy uint64 arithmetic never wraps around -- unlike
# the truncating variant common in MinHash libraries.
_PERM_PRIME = 4_294_967_311  # smallest prime > 2**32
_PERM_DOMAIN = 1 << 32
_PERM_MASK = _PERM_DOMAIN - 1
# Fixed base seed; per-permutation coefficients derive from it deterministically
# via seeded xxhash, so signatures are reproducible with no RNG state to carry.
_BASE_SEED = 0x5170_5550


@dataclass(frozen=True, slots=True)
class NearDedupConfig:
    """Parameters for MinHash near-duplicate detection.

    ``bands`` must divide ``num_perm`` evenly; ``rows`` is the resulting band
    height. The banding S-curve knee sits near ``(1 / bands) ** (1 / rows)``: the
    default 16 bands x 8 rows puts it around Jaccard 0.71, and the
    signature-agreement confirmation at ``threshold`` (default 0.8) keeps only
    genuine high-overlap pairs above that knee.
    """

    shingle_size: int = 5
    num_perm: int = 128
    bands: int = 16
    threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.shingle_size < 1:
            raise ValueError("shingle_size must be >= 1")
        if self.num_perm < 1:
            raise ValueError("num_perm must be >= 1")
        if self.bands < 1:
            raise ValueError("bands must be >= 1")
        if self.num_perm % self.bands != 0:
            raise ValueError(
                f"num_perm ({self.num_perm}) must be divisible by bands ({self.bands}) "
                f"so that bands * rows == num_perm (got remainder {self.num_perm % self.bands})"
            )
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1]; got {self.threshold}")

    @property
    def rows(self) -> int:
        return self.num_perm // self.bands


@dataclass(slots=True)
class NearDedupStats:
    docs_in: int = 0
    docs_out: int = 0
    near_duplicates_dropped: int = 0
    too_short_passed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "docs_in": self.docs_in,
            "docs_out": self.docs_out,
            "near_duplicates_dropped": self.near_duplicates_dropped,
            "too_short_passed": self.too_short_passed,
        }


class NearDeduper:
    """Streaming near-duplicate filter using word-shingle MinHash + LSH banding.

    For each document the word shingles (reusing :func:`word_shingles`, so
    tokenization matches decontamination) are reduced to a ``num_perm``-length
    MinHash signature, split into ``bands`` bands, and looked up in per-band
    tables. A document sharing any band with an earlier kept document is a
    candidate; the candidate is dropped only if the fraction of agreeing
    signature rows -- an unbiased estimate of shingle-set Jaccard -- meets
    ``threshold`` (keep-first-seen, matching :class:`ExactDeduper`).

    The signature-agreement confirmation removes most LSH banding false
    positives: a shared band inspects only ``rows`` of the signature, while
    confirmation checks all ``num_perm`` rows. At the default 128 permutations
    the estimate has a standard error of ~0.035 near the 0.8 threshold --
    precise enough for corpus dedup while keeping the retained state per kept
    document at a fixed ``num_perm * 8`` bytes.

    State is held in memory: the per-band tables plus one signature per kept
    document, roughly 1 KiB per document at the defaults and ~15-25 GiB for the
    full ~5.5M-document merged baseline corpus (sized for a 64 GiB build box).
    Retaining full shingle sets instead was measured at ~73 bytes per shingle,
    i.e. hundreds of GiB at that scale. The filter dedups whatever stream it is
    given -- the baseline corpus build runs it over the merged multi-source
    corpus.
    """

    def __init__(self, config: NearDedupConfig | None = None) -> None:
        self.config = config or NearDedupConfig()
        self.stats = NearDedupStats()
        self._a, self._b = self._permutations(self.config.num_perm)
        self._prime = np.uint64(_PERM_PRIME)
        self._bands: list[dict[bytes, list[int]]] = [{} for _ in range(self.config.bands)]
        self._kept_signatures: list[npt.NDArray[np.uint64]] = []

    @staticmethod
    def _permutations(num_perm: int) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
        """Deterministic per-permutation (a, b) coefficients, both below 2**32."""
        a = np.empty(num_perm, dtype=np.uint64)
        b = np.empty(num_perm, dtype=np.uint64)
        for i in range(num_perm):
            seed = _BASE_SEED + i
            a[i] = 1 + (xxhash.xxh64_intdigest(b"a", seed=seed) % (_PERM_DOMAIN - 1))
            b[i] = xxhash.xxh64_intdigest(b"b", seed=seed) % _PERM_DOMAIN
        return a, b

    def _signature(self, shingles: frozenset[int]) -> npt.NDArray[np.uint64]:
        base = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
        x = base & np.uint64(_PERM_MASK)  # low 32 bits keep a * x + b below 2**64
        permuted = (self._a[:, None] * x[None, :] + self._b[:, None]) % self._prime
        return cast("npt.NDArray[np.uint64]", permuted.min(axis=1))

    def _band_keys(self, signature: npt.NDArray[np.uint64]) -> list[bytes]:
        banded = signature.reshape(self.config.bands, self.config.rows)
        return [banded[i].tobytes() for i in range(self.config.bands)]

    def _candidates(self, band_keys: list[bytes]) -> set[int]:
        found: set[int] = set()
        for band, key in zip(self._bands, band_keys, strict=True):
            found.update(band.get(key, ()))
        return found

    def _confirm_near_duplicate(
        self, signature: npt.NDArray[np.uint64], candidates: set[int]
    ) -> bool:
        threshold = self.config.threshold
        for pos in candidates:
            if float((signature == self._kept_signatures[pos]).mean()) >= threshold:
                return True
        return False

    def _remember(self, signature: npt.NDArray[np.uint64], band_keys: list[bytes]) -> None:
        pos = len(self._kept_signatures)
        self._kept_signatures.append(signature)
        for band, key in zip(self._bands, band_keys, strict=True):
            band.setdefault(key, []).append(pos)

    def filter(self, docs: Iterable[Document]) -> Iterator[Document]:
        for doc in docs:
            self.stats.docs_in += 1
            shingles = frozenset(word_shingles(doc.text, self.config.shingle_size))
            if not shingles:
                # Fewer words than one shingle: cannot be signed, pass through.
                self.stats.too_short_passed += 1
                self.stats.docs_out += 1
                yield doc
                continue
            signature = self._signature(shingles)
            band_keys = self._band_keys(signature)
            candidates = self._candidates(band_keys)
            if candidates and self._confirm_near_duplicate(signature, candidates):
                self.stats.near_duplicates_dropped += 1
                continue
            self._remember(signature, band_keys)
            self.stats.docs_out += 1
            yield doc
