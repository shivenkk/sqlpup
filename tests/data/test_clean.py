import unicodedata

from sqlpup.data.clean import CleanConfig, Cleaner, nonprintable_ratio, normalize_text
from sqlpup.io import Document

LONG_SQL = "SELECT col_a, col_b FROM t WHERE col_a > 10 ORDER BY col_b;\n" * 8


def _doc(text: str, doc_id: str = "d") -> Document:
    return Document(id=doc_id, text=text, source="unit")


def test_keeps_normal_document() -> None:
    cleaner = Cleaner()
    kept = list(cleaner.run([_doc(LONG_SQL)]))
    assert len(kept) == 1
    assert cleaner.stats.kept == 1


def test_drops_short_document() -> None:
    cleaner = Cleaner()
    assert list(cleaner.run([_doc("SELECT 1;")])) == []
    assert cleaner.stats.dropped_short == 1


def test_drops_overlong_document() -> None:
    cleaner = Cleaner(CleanConfig(min_chars=1, max_chars=50))
    assert list(cleaner.run([_doc("x" * 51)])) == []
    assert cleaner.stats.dropped_long == 1


def test_drops_binary_like_document() -> None:
    cleaner = Cleaner(CleanConfig(min_chars=1))
    assert list(cleaner.run([_doc("\x01\x02" * 200)])) == []
    assert cleaner.stats.dropped_nonprintable == 1


def test_normalizes_to_nfc_and_unix_newlines() -> None:
    decomposed = "café\r\nSELECT 1;"
    normalized = normalize_text(decomposed)
    assert "\r" not in normalized
    assert unicodedata.is_normalized("NFC", normalized)


def test_collapses_blank_runs_and_strips_nulls() -> None:
    assert normalize_text("a\n\n\n\n\nb\x00c") == "a\n\nbc"


def test_nonprintable_ratio_ignores_newlines_and_tabs() -> None:
    assert nonprintable_ratio("a\n\tb") == 0.0
    assert nonprintable_ratio("") == 1.0


def test_stats_account_for_every_document() -> None:
    cleaner = Cleaner(CleanConfig(min_chars=5, max_chars=100))
    docs = [_doc("ok" * 10, "a"), _doc("x", "b"), _doc("y" * 200, "c")]
    list(cleaner.run(docs))
    stats = cleaner.stats
    assert stats.kept + stats.dropped_short + stats.dropped_long == 3
