from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from sqlpup.shard import ShardWriter
from sqlpup.train.data import (
    PackedDataLoader,
    TokenStream,
    shard_offsets,
    shuffled_starts,
    window_starts,
)

EOS = 0


def build_stream(tmp_path: Path, tokens: list[int], *, shard_size: int = 10_000) -> TokenStream:
    writer = ShardWriter(tmp_path, eos_id=EOS, shard_size_tokens=shard_size)
    writer.add(tokens)
    writer.close()
    return TokenStream.from_index_file(tmp_path / "index.json")


# --- pure helpers ---------------------------------------------------------


def test_window_starts_non_overlapping() -> None:
    # 7 tokens, seq_len 3 -> windows need seq_len+1=4 tokens; starts stride by
    # seq_len: (7-1)//3 = 2 windows at [0, 3].
    starts = window_starts(7, 3)
    assert list(starts) == [0, 3]


def test_window_starts_empty_when_too_short() -> None:
    assert len(window_starts(3, 4)) == 0


def test_shuffle_deterministic_and_epoch_dependent() -> None:
    starts = window_starts(400, 4)
    a0 = shuffled_starts(starts, seed=7, epoch=0)
    a0_again = shuffled_starts(starts, seed=7, epoch=0)
    a1 = shuffled_starts(starts, seed=7, epoch=1)
    assert np.array_equal(a0, a0_again)  # deterministic per (seed, epoch)
    assert not np.array_equal(a0, a1)  # different epoch reshuffles
    assert np.array_equal(np.sort(a0), starts)  # a permutation of the same starts


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_shard_offsets_disjoint_equal_union(world_size: int) -> None:
    starts = np.arange(10, dtype=np.int64)
    parts = [shard_offsets(starts, r, world_size) for r in range(world_size)]
    per_rank = 10 // world_size
    assert all(len(p) == per_rank for p in parts)  # equal-length (drop-last)
    union = np.concatenate(parts)
    assert len(np.unique(union)) == len(union)  # disjoint across ranks
    assert len(union) == per_rank * world_size  # only the remainder is dropped
    assert set(union.tolist()) <= set(starts.tolist())
    if 10 % world_size == 0:
        assert set(union.tolist()) == set(starts.tolist())  # nothing dropped


# --- TokenStream ----------------------------------------------------------


def test_read_spans_shard_boundary(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, eos_id=7, shard_size_tokens=4)
    writer.add([1, 2, 3])  # -> shard0 = [1, 2, 3, 7]
    writer.add([4, 5])  # -> shard1 = [4, 5, 7]
    writer.close()
    stream = TokenStream.from_index_file(tmp_path / "index.json")
    assert stream.total_tokens == 7
    assert list(stream.read(0, 7)) == [1, 2, 3, 7, 4, 5, 7]
    assert list(stream.read(2, 4)) == [3, 7, 4, 5]  # crosses the 3|4 boundary
    assert list(stream.read(3, 2)) == [7, 4]
    assert stream.read(0, 7).dtype == np.int64


def test_read_out_of_range_rejected(tmp_path: Path) -> None:
    stream = build_stream(tmp_path, [1, 2, 3])  # 4 tokens with eos
    with pytest.raises(IndexError):
        stream.read(2, 5)


# --- PackedDataLoader -----------------------------------------------------


def test_windows_exact_shifted_and_in_shuffled_order(tmp_path: Path) -> None:
    tokens = list(range(1, 20))  # stream = [1..19, 0], 20 tokens
    stream = build_stream(tmp_path, tokens)
    seq_len, seed = 4, 123
    starts = window_starts(stream.total_tokens, seq_len)  # [0, 4, 8, 12]
    order = shard_offsets(shuffled_starts(starts, seed, 0), rank=0, world_size=1)

    loader = PackedDataLoader(stream, seq_len=seq_len, micro_batch_size=1, seed=seed)
    it = iter(loader)
    for expected_start in order:
        x, y = next(it)
        window = stream.read(int(expected_start), seq_len + 1)
        assert torch.equal(x[0], torch.from_numpy(window[:-1]))  # exact window
        assert torch.equal(y[0], torch.from_numpy(window[1:]))  # target
        assert torch.equal(x[0, 1:], y[0, :-1])  # y is x shifted by one


def test_batch_shape_and_dtype(tmp_path: Path) -> None:
    stream = build_stream(tmp_path, list(range(1, 500)))
    loader = PackedDataLoader(stream, seq_len=8, micro_batch_size=3, seed=1)
    x, y = next(iter(loader))
    assert x.shape == (3, 8)
    assert y.shape == (3, 8)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_too_small_corpus_rejected(tmp_path: Path) -> None:
    stream = build_stream(tmp_path, [1, 2, 3])  # only 4 tokens
    with pytest.raises(ValueError, match="too small"):
        PackedDataLoader(stream, seq_len=8, micro_batch_size=1, seed=0)


def test_fast_forward_within_epoch(tmp_path: Path) -> None:
    stream = build_stream(tmp_path, list(range(1, 400)))
    kw = {"seq_len": 4, "micro_batch_size": 2, "seed": 99}
    loader1 = PackedDataLoader(stream, **kw)
    it1 = iter(loader1)
    _ = [next(it1) for _ in range(2)]
    pos = loader1.position()
    future = [next(it1) for _ in range(3)]

    loader2 = PackedDataLoader(stream, **kw, start_epoch=pos[0], start_cursor=pos[1])
    it2 = iter(loader2)
    resumed = [next(it2) for _ in range(3)]
    for (x1, y1), (x2, y2) in zip(future, resumed, strict=True):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


def test_fast_forward_across_epoch_boundary(tmp_path: Path) -> None:
    stream = build_stream(tmp_path, list(range(1, 400)))
    kw = {"seq_len": 4, "micro_batch_size": 2, "seed": 42}
    loader1 = PackedDataLoader(stream, **kw)
    it1 = iter(loader1)
    # Consume more than one epoch so the resume position lands in epoch 1.
    _ = [next(it1) for _ in range(loader1.batches_per_epoch + 2)]
    pos = loader1.position()
    assert pos[0] >= 1
    future = [next(it1) for _ in range(3)]

    loader2 = PackedDataLoader(stream, **kw, start_epoch=pos[0], start_cursor=pos[1])
    it2 = iter(loader2)
    resumed = [next(it2) for _ in range(3)]
    for (x1, y1), (x2, y2) in zip(future, resumed, strict=True):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


def test_rank_sharding_disjoint_batches(tmp_path: Path) -> None:
    stream = build_stream(tmp_path, list(range(1, 400)))
    kw = {"seq_len": 4, "micro_batch_size": 1, "seed": 5}
    r0 = PackedDataLoader(stream, **kw, rank=0, world_size=2)
    r1 = PackedDataLoader(stream, **kw, rank=1, world_size=2)
    # Same epoch, the two ranks draw disjoint window starts (first-token ids).
    it0, it1 = iter(r0), iter(r1)
    firsts0 = {int(next(it0)[0][0, 0]) for _ in range(r0.batches_per_epoch)}
    firsts1 = {int(next(it1)[0][0, 0]) for _ in range(r1.batches_per_epoch)}
    assert firsts0.isdisjoint(firsts1)
