"""Packed memmap batching for pretraining.

The corpus is the concatenation of the uint16 shards named in a
:class:`~sqlpup.shard.ShardIndex`, treated as one virtual token stream. Shards
are memory-mapped (never read whole into RAM), so the working set is only the
windows currently being copied into a batch.

Batches are non-overlapping length-``seq_len`` windows of that stream: window
``k`` covers tokens ``[k*seq_len, k*seq_len + seq_len + 1)`` and yields
``x = window[:-1]``, ``y = window[1:]`` (next-token targets). Adjacent windows
therefore share only the single boundary token (the last target of window ``k``
is the first input of window ``k+1``) -- the standard nanoGPT packed layout the
design spec calls for. Document boundaries are whatever the shard writer packed
(EOS-separated); no resegmentation happens here.

Shuffling and rank-sharding are factored into the module-level pure functions
:func:`window_starts`, :func:`shuffled_starts`, and :func:`shard_offsets` so the
sampling logic is unit-testable without a running loop or ``torch.distributed``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from sqlpup.shard import ShardIndex, read_shard

Int64Array = npt.NDArray[np.int64]


def window_starts(total_tokens: int, seq_len: int) -> Int64Array:
    """Stream offsets of the non-overlapping length-``seq_len`` windows.

    Each window reads ``seq_len + 1`` tokens (inputs + shifted target), so the
    last usable start is ``total_tokens - seq_len - 1``; there are
    ``(total_tokens - 1) // seq_len`` windows.
    """
    num_windows = max(0, (total_tokens - 1) // seq_len)
    return np.arange(num_windows, dtype=np.int64) * seq_len


def shuffled_starts(starts: Int64Array, seed: int, epoch: int) -> Int64Array:
    """Return ``starts`` permuted deterministically for one epoch.

    The RNG is derived purely from ``seed + epoch`` (a fresh generator each
    epoch), so a resumed run reproduces any epoch's order from ``(seed, epoch)``
    alone -- no long-lived generator state to checkpoint.
    """
    rng = np.random.default_rng(seed + epoch)
    return starts[rng.permutation(len(starts))]


def shard_offsets(starts: Int64Array, rank: int, world_size: int) -> Int64Array:
    """Take rank ``rank``'s share of ``starts`` under DDP (strided, drop-last).

    Rank ``r`` of ``world_size`` takes ``starts[r::world_size]`` truncated to the
    common per-rank length ``len(starts) // world_size``, so every rank steps the
    same number of times and stays in lockstep for gradient all-reduce.
    """
    per_rank = len(starts) // world_size
    return starts[rank::world_size][:per_rank]


class TokenStream:
    """The shards of one :class:`ShardIndex` as a single virtual token stream."""

    def __init__(self, index: ShardIndex, shard_dir: Path) -> None:
        self._mmaps = tuple(read_shard(shard_dir / s.file) for s in index.shards)
        lengths = [len(m) for m in self._mmaps]
        # _cum[i] is the stream offset where shard i starts; _cum[-1] is total.
        self._cum = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        self._total = int(self._cum[-1])

    @classmethod
    def from_index_file(cls, path: Path) -> TokenStream:
        """Load the index at ``path`` and memory-map its sibling shard files."""
        return cls(ShardIndex.load(path), path.parent)

    @property
    def total_tokens(self) -> int:
        return self._total

    def read(self, start: int, length: int) -> Int64Array:
        """Return ``length`` tokens from ``start``, stitching across shards."""
        if start < 0 or length < 0 or start + length > self._total:
            raise IndexError(f"window [{start}, {start + length}) out of range [0, {self._total})")
        out = np.empty(length, dtype=np.int64)
        pos = 0
        shard_i = int(np.searchsorted(self._cum, start, side="right")) - 1
        local = start - int(self._cum[shard_i])
        while pos < length:
            mmap = self._mmaps[shard_i]
            take = min(length - pos, len(mmap) - local)
            out[pos : pos + take] = mmap[local : local + take]  # uint16 -> int64
            pos += take
            shard_i += 1
            local = 0
        return out


class PackedDataLoader:
    """Stateful, resumable iterator of packed ``(x, y)`` int64 CPU tensors.

    Iterating never stops: at the end of an epoch it reshuffles with the next
    epoch seed and continues, so training just pulls ``total_steps *
    grad_accum_steps`` batches. The position ``(epoch, cursor)`` -- ``cursor``
    being the next window index within the epoch -- is all that is needed to
    resume: reconstruct with ``start_epoch``/``start_cursor`` (or call
    :meth:`seek`) and the next batch matches uninterrupted iteration exactly.
    """

    def __init__(
        self,
        stream: TokenStream,
        seq_len: int,
        micro_batch_size: int,
        seed: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        start_epoch: int = 0,
        start_cursor: int = 0,
    ) -> None:
        self.stream = stream
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self._all_starts = window_starts(stream.total_tokens, seq_len)
        # Per-rank window count is constant across epochs (only order changes).
        self._windows_per_epoch = len(self._all_starts) // world_size
        if self._windows_per_epoch < micro_batch_size:
            raise ValueError(
                f"corpus too small: {len(self._all_starts)} windows over world_size "
                f"{world_size} gives {self._windows_per_epoch} per rank, "
                f"< micro_batch_size {micro_batch_size}"
            )
        self.epoch = start_epoch
        self.cursor = start_cursor
        self._epoch_starts = self._starts_for(start_epoch)

    @property
    def batches_per_epoch(self) -> int:
        """Full micro-batches drawn per epoch per rank (remainder dropped)."""
        return self._windows_per_epoch // self.micro_batch_size

    def position(self) -> tuple[int, int]:
        """Current resumable position ``(epoch, cursor)`` (next batch to emit)."""
        return (self.epoch, self.cursor)

    def seek(self, epoch: int, cursor: int) -> None:
        """Jump to a saved ``(epoch, cursor)`` position for resume."""
        self.epoch = epoch
        self.cursor = cursor
        self._epoch_starts = self._starts_for(epoch)

    def _starts_for(self, epoch: int) -> Int64Array:
        shuffled = shuffled_starts(self._all_starts, self.seed, epoch)
        return shard_offsets(shuffled, self.rank, self.world_size)

    def __iter__(self) -> PackedDataLoader:
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self.cursor + self.micro_batch_size > self._windows_per_epoch:
            # Not enough windows left for a full batch: advance to the next epoch
            # (drop-last keeps every rank's step count identical).
            self.epoch += 1
            self._epoch_starts = self._starts_for(self.epoch)
            self.cursor = 0
        batch_starts = self._epoch_starts[self.cursor : self.cursor + self.micro_batch_size]
        x = np.empty((self.micro_batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((self.micro_batch_size, self.seq_len), dtype=np.int64)
        for i, start in enumerate(batch_starts):
            window = self.stream.read(int(start), self.seq_len + 1)
            x[i] = window[:-1]
            y[i] = window[1:]
        self.cursor += self.micro_batch_size
        return torch.from_numpy(x), torch.from_numpy(y)
