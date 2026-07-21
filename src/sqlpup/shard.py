"""Fixed-dtype token shards for the pretraining dataloader.

Token ids are stored as raw little-endian uint16 arrays (one ``.bin`` file per
``shard_size_tokens``) with an ``index.json`` describing the set. Documents
are separated by the EOS token id appended after every sequence. uint16 caps
the vocabulary at 65,536 entries, which the 32k tokenizer fits with headroom
while halving disk and dataloader bandwidth versus int32.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

UINT16_MAX = 65_535


@dataclass(frozen=True, slots=True)
class ShardInfo:
    file: str
    tokens: int


@dataclass(frozen=True, slots=True)
class ShardIndex:
    dtype: str
    eos_id: int
    total_tokens: int
    shards: tuple[ShardInfo, ...]

    def save(self, path: Path) -> None:
        payload = {
            "dtype": self.dtype,
            "eos_id": self.eos_id,
            "total_tokens": self.total_tokens,
            "shards": [{"file": s.file, "tokens": s.tokens} for s in self.shards],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ShardIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            dtype=str(payload["dtype"]),
            eos_id=int(payload["eos_id"]),
            total_tokens=int(payload["total_tokens"]),
            shards=tuple(
                ShardInfo(file=str(s["file"]), tokens=int(s["tokens"])) for s in payload["shards"]
            ),
        )


class ShardWriter:
    """Append token sequences and roll ``.bin`` files at a fixed token count."""

    def __init__(
        self,
        out_dir: Path,
        eos_id: int,
        shard_size_tokens: int = 100_000_000,
        prefix: str = "shard",
    ) -> None:
        if not 0 <= eos_id <= UINT16_MAX:
            raise ValueError(f"eos_id {eos_id} does not fit uint16")
        if shard_size_tokens < 1:
            raise ValueError("shard_size_tokens must be >= 1")
        self.out_dir = out_dir
        self.eos_id = eos_id
        self.shard_size_tokens = shard_size_tokens
        self.prefix = prefix
        self._buffer: list[int] = []
        self._shards: list[ShardInfo] = []
        self._closed = False
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, token_ids: Sequence[int]) -> None:
        """Append one document's token ids; EOS is added automatically."""
        if self._closed:
            raise RuntimeError("writer is closed")
        if len(token_ids) > 0:
            array = np.asarray(token_ids)
            low = int(array.min())
            high = int(array.max())
            if low < 0 or high > UINT16_MAX:
                raise ValueError(f"token id out of uint16 range: min={low}, max={high}")
            self._buffer.extend(int(t) for t in token_ids)
        self._buffer.append(self.eos_id)
        while len(self._buffer) >= self.shard_size_tokens:
            self._flush(self.shard_size_tokens)

    def close(self) -> ShardIndex:
        """Flush the remainder and write ``index.json``; returns the index."""
        if self._closed:
            raise RuntimeError("writer is closed")
        if self._buffer:
            self._flush(len(self._buffer))
        self._closed = True
        index = ShardIndex(
            dtype="uint16",
            eos_id=self.eos_id,
            total_tokens=sum(s.tokens for s in self._shards),
            shards=tuple(self._shards),
        )
        index.save(self.out_dir / "index.json")
        return index

    def _flush(self, count: int) -> None:
        chunk, self._buffer = self._buffer[:count], self._buffer[count:]
        name = f"{self.prefix}_{len(self._shards):05d}.bin"
        np.asarray(chunk, dtype=np.uint16).tofile(self.out_dir / name)
        self._shards.append(ShardInfo(file=name, tokens=len(chunk)))


def read_shard(path: Path) -> npt.NDArray[np.uint16]:
    """Memory-map one shard file."""
    return np.memmap(path, dtype=np.uint16, mode="r")
