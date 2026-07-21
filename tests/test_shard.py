from pathlib import Path

import numpy as np
import pytest

from sqlpup.shard import ShardIndex, ShardWriter, read_shard

EOS = 7


def _read_all(out_dir: Path, index: ShardIndex) -> list[int]:
    tokens: list[int] = []
    for shard in index.shards:
        tokens.extend(int(t) for t in read_shard(out_dir / shard.file))
    return tokens


def test_roundtrip_with_eos_separators(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, eos_id=EOS)
    writer.add([1, 2, 3])
    writer.add([4, 5])
    index = writer.close()
    assert _read_all(tmp_path, index) == [1, 2, 3, EOS, 4, 5, EOS]
    assert index.total_tokens == 7


def test_rollover_at_shard_boundary(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, eos_id=EOS, shard_size_tokens=4)
    writer.add([1, 2, 3])  # 4 tokens with eos -> exactly one shard
    writer.add([4, 5])  # 3 tokens -> second shard on close
    index = writer.close()
    assert [s.tokens for s in index.shards] == [4, 3]
    assert index.total_tokens == 7


def test_index_json_roundtrip(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, eos_id=EOS)
    writer.add([1])
    index = writer.close()
    loaded = ShardIndex.load(tmp_path / "index.json")
    assert loaded == index


def test_out_of_range_token_rejected(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, eos_id=EOS)
    with pytest.raises(ValueError, match="uint16"):
        writer.add([70_000])
    with pytest.raises(ValueError, match="uint16"):
        writer.add([-1])


def test_shard_files_are_uint16(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, eos_id=EOS)
    writer.add([65_535])
    index = writer.close()
    data = read_shard(tmp_path / index.shards[0].file)
    assert data.dtype == np.uint16


def test_closed_writer_rejects_use(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, eos_id=EOS)
    writer.close()
    with pytest.raises(RuntimeError):
        writer.add([1])
