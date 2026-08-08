"""Unit tests for the smoke-train script's importable core (torch-free).

The full smoke run itself is deliberately *not* driven from pytest -- CI runs it
as its own step and the suite stays fast. Here we only exercise the synthetic
shard generation: shape and determinism.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from sqlpup.shard import ShardIndex, read_shard


def _load_smoke() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_train.py"
    spec = importlib.util.spec_from_file_location("smoke_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke()


def test_build_pattern_is_deterministic_in_seed() -> None:
    assert smoke.build_pattern(256, 16, 0) == smoke.build_pattern(256, 16, 0)
    assert smoke.build_pattern(256, 16, 0) != smoke.build_pattern(256, 16, 1)
    pattern = smoke.build_pattern(256, 16, 0)
    assert len(pattern) == 16
    assert len(set(pattern)) == 16  # distinct token ids
    assert all(0 <= t < 255 for t in pattern)  # strictly below the reserved EOS id


def test_build_pattern_rejects_period_not_below_vocab() -> None:
    with pytest.raises(ValueError):
        smoke.build_pattern(16, 16, 0)


def test_write_synthetic_shards_shape_and_determinism(tmp_path: Path) -> None:
    index_a: ShardIndex = smoke.write_synthetic_shards(
        tmp_path / "a", documents=32, repeats=8, seed=0
    )
    index_b: ShardIndex = smoke.write_synthetic_shards(
        tmp_path / "b", documents=32, repeats=8, seed=0
    )
    # 32 docs * (16-token cycle * 8 repeats + 1 EOS) = 32 * 129 = 4128 tokens.
    assert index_a.total_tokens == 32 * (16 * 8 + 1)
    assert index_a.dtype == "uint16"
    assert index_a.eos_id == 255
    # Same seed -> byte-identical shard files.
    assert index_a.total_tokens == index_b.total_tokens
    bytes_a = read_shard(tmp_path / "a" / index_a.shards[0].file).tobytes()
    bytes_b = read_shard(tmp_path / "b" / index_b.shards[0].file).tobytes()
    assert bytes_a == bytes_b
