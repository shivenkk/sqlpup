import pytest

from sqlpup.data.decontaminate import word_shingles
from sqlpup.data.dedup import ExactDeduper, NearDedupConfig, NearDeduper
from sqlpup.io import Document


def _doc(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, text=text, source="unit")


def test_drops_exact_duplicate_keeps_first() -> None:
    deduper = ExactDeduper()
    docs = [_doc("a", "SELECT 1;"), _doc("b", "SELECT 1;"), _doc("c", "SELECT 2;")]
    kept = list(deduper.filter(docs))
    assert [d.id for d in kept] == ["a", "c"]
    assert deduper.stats.as_dict() == {"seen": 3, "unique": 2, "dropped": 1}


def test_whitespace_variants_collapse_by_default() -> None:
    deduper = ExactDeduper()
    docs = [_doc("a", "SELECT  1 ;"), _doc("b", " SELECT 1 ;\n")]
    assert [d.id for d in deduper.filter(docs)] == ["a"]


def test_whitespace_sensitive_mode_keeps_variants() -> None:
    deduper = ExactDeduper(normalize_whitespace=False)
    docs = [_doc("a", "SELECT  1;"), _doc("b", "SELECT 1;")]
    assert [d.id for d in deduper.filter(docs)] == ["a", "b"]


# --- near-duplicate filtering (MinHash + LSH) -------------------------------

# A realistic-length document (~180 words -> ~176 five-word shingles) so the
# 128-permutation signature and 16-band LSH banding are genuinely exercised,
# not a three-word toy that would collapse the S-curve.
LONG_TEXT = (
    "Modern data platforms ingest information from many different systems "
    "before analysts can ask meaningful questions about it. A typical warehouse "
    "first collects raw records from transactional databases, application logs, "
    "and third party feeds, then normalizes every field into a consistent schema. "
    "Engineers write transformation jobs that clean malformed rows, remove "
    "duplicate entries, and enrich records with reference tables maintained by "
    "other teams. Once the curated tables are ready, a query engine lets people "
    "express complex questions using structured language, joining several tables "
    "together and aggregating millions of rows in seconds. Good documentation "
    "matters enormously here, because a column whose meaning is unclear will "
    "eventually be misinterpreted by someone building an important report. "
    "Careful teams therefore invest in catalogs that describe each dataset, its "
    "owner, its refresh cadence, and any known quality caveats. When a pipeline "
    "breaks overnight, on call engineers rely on detailed logs and lineage graphs "
    "to trace exactly which upstream change introduced the problem. Over time "
    "these practices compound, turning a fragile collection of scripts into a "
    "dependable foundation that the entire company can trust for reporting, "
    "forecasting, experimentation, and everyday operational decisions across many "
    "departments."
)

# The same document with three well-separated single-occurrence words swapped;
# Jaccard stays above the 0.8 default threshold (verified in the test below).
NEAR_TEXT = (
    LONG_TEXT.replace("enormously", "greatly")
    .replace("overnight", "unexpectedly")
    .replace("fragile", "brittle")
)

# An unrelated document of comparable length that shares no five-word shingle.
DISTINCT_TEXT = (
    "The narrow mountain trail climbed steeply through dense pine forest before "
    "emerging onto a windswept ridge high above the tree line. Hikers paused to "
    "refill their bottles from a cold spring, tightened worn boot laces, and "
    "studied a folded paper map against the fading afternoon light. Somewhere "
    "below, a river carved its patient path between granite boulders polished "
    "smooth by centuries of snowmelt. Marmots whistled sharp warnings from sunlit "
    "rocks while an eagle circled lazily on a rising thermal, scanning the meadow "
    "for careless prey. The air grew thin and noticeably colder as the group "
    "gained elevation, forcing slow deliberate steps and frequent measured "
    "breaths. By dusk they reached a sheltered alpine basin, pitched two small "
    "tents beside a quiet tarn, and cooked a simple warm meal while the first "
    "bright stars appeared over the jagged silhouette of the distant summit ahead "
    "of them tomorrow. Nobody spoke very much, content simply to breathe the crisp "
    "silence and watch the daylight drain slowly from the surrounding frozen peaks."
)


def _jaccard(a: str, b: str, n: int) -> float:
    left, right = word_shingles(a, n), word_shingles(b, n)
    return len(left & right) / len(left | right)


def test_near_dedup_fixtures_have_realistic_shape() -> None:
    assert len(LONG_TEXT.split()) >= 150
    assert len(DISTINCT_TEXT.split()) >= 150
    # the near-duplicate really is a near-duplicate, not identical and not distinct
    assert 0.8 < _jaccard(LONG_TEXT, NEAR_TEXT, 5) < 1.0
    assert _jaccard(LONG_TEXT, DISTINCT_TEXT, 5) < 0.05


def test_near_dedup_drops_identical_document_keeps_first() -> None:
    deduper = NearDeduper()
    docs = [_doc("a", LONG_TEXT), _doc("b", LONG_TEXT), _doc("c", DISTINCT_TEXT)]
    kept = list(deduper.filter(docs))
    assert [d.id for d in kept] == ["a", "c"]
    assert deduper.stats.as_dict() == {
        "docs_in": 3,
        "docs_out": 2,
        "near_duplicates_dropped": 1,
        "too_short_passed": 0,
    }


def test_near_dedup_drops_realistic_near_duplicate() -> None:
    deduper = NearDeduper()
    kept = list(deduper.filter([_doc("orig", LONG_TEXT), _doc("near", NEAR_TEXT)]))
    assert [d.id for d in kept] == ["orig"]
    assert deduper.stats.near_duplicates_dropped == 1


def test_near_dedup_keeps_distinct_documents() -> None:
    deduper = NearDeduper()
    kept = list(deduper.filter([_doc("a", LONG_TEXT), _doc("b", DISTINCT_TEXT)]))
    assert [d.id for d in kept] == ["a", "b"]
    assert deduper.stats.near_duplicates_dropped == 0


def test_near_dedup_is_deterministic() -> None:
    docs = [
        _doc("a", LONG_TEXT),
        _doc("b", NEAR_TEXT),
        _doc("c", DISTINCT_TEXT),
        _doc("short", "select one row"),
    ]
    first = NearDeduper()
    second = NearDeduper()
    kept_first = [d.id for d in first.filter(docs)]
    kept_second = [d.id for d in second.filter(docs)]
    assert kept_first == kept_second
    assert first.stats.as_dict() == second.stats.as_dict()


def test_near_dedup_passes_short_documents_through() -> None:
    # With shingle_size 5, a doc of fewer than five words cannot be signed and
    # is passed through unfiltered, counted separately, never dropped.
    docs = [
        _doc("s1", "select one row"),
        _doc("big", LONG_TEXT),
        _doc("s2", "another tiny doc"),
    ]
    deduper = NearDeduper()
    kept = list(deduper.filter(docs))
    assert [d.id for d in kept] == ["s1", "big", "s2"]
    assert deduper.stats.too_short_passed == 2
    assert deduper.stats.near_duplicates_dropped == 0
    assert deduper.stats.docs_out == 3


def test_near_dedup_retains_fixed_size_state_per_kept_doc() -> None:
    # The memory contract that lets near-dedup run over the full merged corpus:
    # a kept document leaves behind only its num_perm-length MinHash signature
    # (plus band-table entries), never its full shingle set, so retained state
    # is independent of document length.
    config = NearDedupConfig()
    deduper = NearDeduper(config)
    kept = list(deduper.filter([_doc("a", LONG_TEXT), _doc("b", DISTINCT_TEXT)]))
    assert len(kept) == 2
    assert not hasattr(deduper, "_kept_shingles")
    assert len(deduper._kept_signatures) == 2
    assert all(sig.shape == (config.num_perm,) for sig in deduper._kept_signatures)


def test_near_dedup_config_rejects_indivisible_banding() -> None:
    with pytest.raises(ValueError, match="num_perm"):
        NearDedupConfig(num_perm=128, bands=15)


def test_near_dedup_config_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        NearDedupConfig(threshold=1.5)
