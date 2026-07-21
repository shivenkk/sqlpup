import json
from pathlib import Path

import pytest

from sqlpup.data.decontaminate import (
    Decontaminator,
    NgramIndex,
    load_eval_texts,
    word_shingles,
)
from sqlpup.io import Document

EVAL_TEXT = "how many students scored above ninety in the final math exam last spring semester"


def _doc(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, text=text, source="unit")


def test_document_containing_eval_ngram_is_dropped() -> None:
    index = NgramIndex.build([EVAL_TEXT], n=13)
    contaminated = _doc("bad", f"Random prefix. {EVAL_TEXT} Random suffix.")
    clean = _doc("ok", "SELECT name FROM students WHERE score > 90;")
    decon = Decontaminator(index)
    kept = list(decon.filter([contaminated, clean]))
    assert [d.id for d in kept] == ["ok"]
    assert decon.stats.as_dict() == {"kept": 1, "dropped": 1}


def test_partial_overlap_below_ngram_length_is_kept() -> None:
    index = NgramIndex.build([EVAL_TEXT], n=13)
    twelve_words = " ".join(EVAL_TEXT.split()[:12])
    doc = _doc("ok", f"{twelve_words} and now completely different tail words here")
    assert [d.id for d in Decontaminator(index).filter([doc])] == ["ok"]


def test_short_eval_text_contributes_no_shingles() -> None:
    assert word_shingles("only five words right here", 13) == set()


def test_shingles_ignore_case_and_punctuation() -> None:
    a = word_shingles(EVAL_TEXT, 13)
    b = word_shingles(EVAL_TEXT.upper().replace(" ", ",  "), 13)
    assert a == b


def test_index_save_load_roundtrip(tmp_path: Path) -> None:
    index = NgramIndex.build([EVAL_TEXT], n=13)
    path = tmp_path / "index.json"
    index.save(path)
    loaded = NgramIndex.load(path)
    assert loaded == index


def test_threshold_validation() -> None:
    index = NgramIndex.build([], n=13)
    with pytest.raises(ValueError, match=">= 1"):
        Decontaminator(index, threshold=0)


def test_load_eval_texts_reads_bird_and_spider_shapes(tmp_path: Path) -> None:
    bird = tmp_path / "bird.json"
    bird.write_text(json.dumps([{"question": "How many users?", "SQL": "SELECT COUNT(*) FROM u"}]))
    spider = tmp_path / "spider.json"
    spider.write_text(json.dumps([{"question": "List names", "query": "SELECT name FROM t"}]))
    texts = load_eval_texts([bird, spider])
    assert texts == [
        "How many users? SELECT COUNT(*) FROM u",
        "List names SELECT name FROM t",
    ]


def test_load_eval_texts_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="JSON list"):
        load_eval_texts([path])
