"""Document model and JSONL(.zst) corpus I/O.

Every pipeline stage consumes and produces streams of ``Document`` records
serialized as JSON Lines, zstd-compressed when the path ends in ``.zst``.
Keeping the interchange format this simple makes each stage independently
runnable, resumable, and testable offline.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, cast

import zstandard


@dataclass(frozen=True, slots=True)
class Document:
    """A single training document flowing through the data pipeline."""

    id: str
    text: str
    source: str
    meta: dict[str, str] = field(default_factory=dict)


def _open_write(path: Path) -> IO[bytes]:
    raw = path.open("wb")
    if path.suffix == ".zst":
        compressor = zstandard.ZstdCompressor(level=10)
        return cast("IO[bytes]", compressor.stream_writer(raw, closefd=True))
    return raw


def _open_read(path: Path) -> IO[bytes]:
    raw = path.open("rb")
    if path.suffix == ".zst":
        decompressor = zstandard.ZstdDecompressor()
        return cast("IO[bytes]", decompressor.stream_reader(raw, closefd=True))
    return raw


def write_documents(path: Path, docs: Iterable[Document]) -> int:
    """Write documents to ``path`` as JSONL(.zst); return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_write(path) as handle:
        for doc in docs:
            line = json.dumps(asdict(doc), ensure_ascii=False) + "\n"
            handle.write(line.encode("utf-8"))
            count += 1
    return count


def read_documents(path: Path) -> Iterator[Document]:
    """Stream documents back from a file written by :func:`write_documents`."""
    with _open_read(path) as handle:
        for line in io.TextIOWrapper(handle, encoding="utf-8"):
            if not line.strip():
                continue
            raw = json.loads(line)
            yield Document(
                id=raw["id"],
                text=raw["text"],
                source=raw["source"],
                meta=dict(raw.get("meta", {})),
            )
