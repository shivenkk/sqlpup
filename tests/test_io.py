from pathlib import Path

import pytest

from sqlpup.io import Document, read_documents, write_documents

DOCS = [
    Document(id="a", text="SELECT 1;", source="unit", meta={"k": "v"}),
    Document(id="b", text="two\nlines", source="unit"),
]


def test_roundtrip_plain(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    assert write_documents(path, DOCS) == 2
    assert list(read_documents(path)) == DOCS


def test_roundtrip_zst(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl.zst"
    assert write_documents(path, DOCS) == 2
    assert list(read_documents(path)) == DOCS


def test_unicode_preserved(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl.zst"
    doc = Document(id="u", text="naïve — データ ✓", source="unit")
    write_documents(path, [doc])
    assert next(iter(read_documents(path))).text == "naïve — データ ✓"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(read_documents(tmp_path / "absent.jsonl"))


def test_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "docs.jsonl"
    write_documents(path, DOCS)
    assert path.exists()
