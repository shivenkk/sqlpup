import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from sqlpup.config import EvalSetsConfig, EvalSourceConfig
from sqlpup.data import download
from sqlpup.data.decontaminate import Decontaminator, NgramIndex
from sqlpup.data.eval_sets import build_eval_index, extract_field_texts, read_member
from sqlpup.io import Document

FIXTURES = Path(__file__).parent / "fixtures"
BIRD_FIXTURE = FIXTURES / "bird_dev_sample.json"
SPIDER_FIXTURE = FIXTURES / "spider_dev_sample.json"

BIRD_URL = "http://eval.test/bird_dev.zip"
SPIDER_URL = "http://eval.test/spider_data.zip"


def _load_examples(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


# --- field extraction -------------------------------------------------------


def test_extract_field_texts_emits_each_field_and_skips_empty_evidence() -> None:
    examples = _load_examples(BIRD_FIXTURE)
    texts = extract_field_texts(examples, ["question", "SQL", "evidence"])
    # 4 questions + 4 SQL + 2 non-empty evidence (records 1 and 3 lack evidence)
    assert len(texts) == 10
    assert examples[0]["question"] in texts
    assert examples[0]["SQL"] in texts
    assert examples[0]["evidence"] in texts
    # the empty-string evidence and the absent evidence never become entries
    assert "" not in texts
    non_empty_evidence = [e["evidence"] for e in examples if e.get("evidence")]
    assert len(non_empty_evidence) == 2


def test_extract_field_texts_skips_missing_and_non_string_fields() -> None:
    examples: list[dict[str, Any]] = [
        {"question": "a proper natural language question written out in words", "SQL": 12345},
        {"question": "   ", "SQL": "SELECT 1", "evidence": None},
    ]
    texts = extract_field_texts(examples, ["question", "SQL", "evidence"])
    # int SQL, None evidence, whitespace-only question, and missing evidence all skipped
    assert texts == ["a proper natural language question written out in words", "SELECT 1"]


# --- zip member extraction --------------------------------------------------


def test_read_member_extracts_named_json_from_zip(tmp_path: Path) -> None:
    payload = BIRD_FIXTURE.read_bytes()
    archive = tmp_path / "bird.zip"
    archive.write_bytes(
        _zip_bytes(
            {
                "dev_20240627/dev.json": payload,
                "__MACOSX/dev_20240627/._dev.json": b"resource-fork junk",
            }
        )
    )
    got = read_member(archive, "dev_20240627/dev.json")
    assert json.loads(got) == json.loads(payload)


def test_read_member_reads_whole_file_when_member_is_none(tmp_path: Path) -> None:
    direct = tmp_path / "spider.json"
    direct.write_bytes(SPIDER_FIXTURE.read_bytes())
    assert read_member(direct, None) == SPIDER_FIXTURE.read_bytes()


def test_read_member_missing_member_raises(tmp_path: Path) -> None:
    archive = tmp_path / "z.zip"
    archive.write_bytes(_zip_bytes({"real.json": b"[]"}))
    with pytest.raises(KeyError, match=r"nope\.json"):
        read_member(archive, "nope.json")


# --- index build + report ---------------------------------------------------


@pytest.fixture
def offline_fetch(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Serve the real fixtures in place of a network download, counting calls."""
    payloads = {
        BIRD_URL: _zip_bytes({"dev_20240627/dev.json": BIRD_FIXTURE.read_bytes()}),
        SPIDER_URL: _zip_bytes(
            {
                "spider_data/dev.json": SPIDER_FIXTURE.read_bytes(),
                "spider_data/test.json": SPIDER_FIXTURE.read_bytes(),
                "__MACOSX/spider_data/._dev.json": b"resource-fork junk",
            }
        ),
    }
    calls = {"count": 0}

    def fake_download(url: str, dest: Path) -> None:
        calls["count"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payloads[url])

    monkeypatch.setattr(download, "_download_artifact", fake_download)
    return calls


def _config() -> EvalSetsConfig:
    return EvalSetsConfig(
        sources=(
            EvalSourceConfig(
                name="bird_dev",
                url=BIRD_URL,
                fields=("question", "SQL", "evidence"),
                archive_member="dev_20240627/dev.json",
                expected_examples=4,
            ),
            EvalSourceConfig(
                name="spider_dev",
                url=SPIDER_URL,
                fields=("question", "query"),
                archive_member="spider_data/dev.json",
                expected_examples=4,
            ),
            EvalSourceConfig(
                name="spider_test",
                url=SPIDER_URL,
                fields=("question", "query"),
                archive_member="spider_data/test.json",
                expected_examples=4,
            ),
        )
    )


def test_build_eval_index_writes_report_and_caches_downloads(
    tmp_path: Path, offline_fetch: dict[str, int]
) -> None:
    index_path = tmp_path / "artifacts" / "decontam" / "eval_index.json"
    report_path = tmp_path / "artifacts" / "decontam" / "eval_index_report.json"
    report = build_eval_index(_config(), tmp_path / "eval", index_path, report_path)

    # two unique URLs -> two downloads, though spider_data.zip feeds two sources
    assert offline_fetch["count"] == 2
    assert index_path.exists()
    assert report_path.exists()

    assert report["n"] == 13
    assert report["total_shingles"] > 0
    assert NgramIndex.load(index_path).n == 13

    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk == report

    by_name = {s["name"]: s for s in report["sources"]}
    assert by_name["bird_dev"]["examples"] == 4
    assert by_name["bird_dev"]["text_entries"] == 10  # 4 question + 4 SQL + 2 evidence
    assert by_name["spider_dev"]["text_entries"] == 8  # 4 question + 4 query
    assert by_name["spider_test"]["text_entries"] == 8
    assert by_name["bird_dev"]["expected_examples"] == 4

    # provenance: 64-hex sha256 per source; a shared archive shares its digest
    for src in report["sources"]:
        assert len(src["sha256"]) == 64
        int(src["sha256"], 16)
    assert by_name["spider_dev"]["sha256"] == by_name["spider_test"]["sha256"]
    assert by_name["bird_dev"]["sha256"] != by_name["spider_dev"]["sha256"]

    parsed = datetime.fromisoformat(report["generated_at"])
    assert parsed.tzinfo is not None

    # re-running reuses the cached archives: no further downloads
    build_eval_index(_config(), tmp_path / "eval", index_path, report_path)
    assert offline_fetch["count"] == 2


def test_built_index_drops_contaminated_document(
    tmp_path: Path, offline_fetch: dict[str, int]
) -> None:
    index_path = tmp_path / "eval_index.json"
    build_eval_index(_config(), tmp_path / "eval", index_path, tmp_path / "report.json")

    index = NgramIndex.load(index_path)
    question = _load_examples(BIRD_FIXTURE)[0]["question"]
    contaminated = Document(id="bad", text=f"Intro text. {question} Trailing text.", source="unit")
    clean = Document(id="ok", text="SELECT 1;", source="unit")

    decon = Decontaminator(index)
    kept = list(decon.filter([contaminated, clean]))
    assert [d.id for d in kept] == ["ok"]
    assert decon.stats.as_dict() == {"kept": 1, "dropped": 1}


def test_index_build_is_deterministic(tmp_path: Path, offline_fetch: dict[str, int]) -> None:
    a_index = tmp_path / "a.json"
    b_index = tmp_path / "b.json"
    first = build_eval_index(_config(), tmp_path / "eval", a_index, tmp_path / "a_report.json")
    second = build_eval_index(_config(), tmp_path / "eval", b_index, tmp_path / "b_report.json")
    assert first["total_shingles"] == second["total_shingles"]
    assert NgramIndex.load(a_index) == NgramIndex.load(b_index)
    # the saved index file is byte-identical across builds (published artifact)
    assert a_index.read_bytes() == b_index.read_bytes()


def _bird_only(expected: int | None) -> EvalSetsConfig:
    return EvalSetsConfig(
        sources=(
            EvalSourceConfig(
                name="bird_dev",
                url=BIRD_URL,
                fields=("question", "SQL", "evidence"),
                archive_member="dev_20240627/dev.json",
                expected_examples=expected,
            ),
        )
    )


def test_build_eval_index_fails_loud_on_count_mismatch(
    tmp_path: Path, offline_fetch: dict[str, int]
) -> None:
    # the bird fixture holds 4 examples; a config claiming 5 must fail loud
    index_path = tmp_path / "i.json"
    with pytest.raises(ValueError, match="bird_dev"):
        build_eval_index(_bird_only(5), tmp_path / "eval", index_path, tmp_path / "r.json")
    # a bad artifact must not be written when the count check fails
    assert not index_path.exists()


def test_build_eval_index_skips_count_check_when_expected_unset(
    tmp_path: Path, offline_fetch: dict[str, int]
) -> None:
    report = build_eval_index(
        _bird_only(None), tmp_path / "eval", tmp_path / "i.json", tmp_path / "r.json"
    )
    assert report["sources"][0]["examples"] == 4
    assert report["sources"][0]["expected_examples"] is None
