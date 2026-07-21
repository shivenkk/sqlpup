import gzip
import json
from collections.abc import Iterator, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from sqlpup.config import SourceConfig
from sqlpup.data.download import download_source, record_to_document
from sqlpup.io import read_documents


def _cfg(**overrides: Any) -> SourceConfig:
    base: dict[str, Any] = {
        "name": "unit",
        "hf_path": "org/dataset",
        "hf_name": None,
        "split": "train",
        "license": "MIT",
        "target_tokens": 1_000_000,
        "text_field": "text",
        "doc_cap": None,
    }
    base.update(overrides)
    return SourceConfig(**base)


def _reader(records: list[dict[str, Any]]) -> Any:
    def read(_: SourceConfig) -> Iterator[Mapping[str, Any]]:
        yield from records

    return read


def test_writes_documents_and_manifest(tmp_path: Path) -> None:
    records = [{"text": "SELECT 1;" * 20}, {"text": "SELECT 2;" * 20}]
    result = download_source(_cfg(), tmp_path, reader=_reader(records))
    assert result.docs == 2
    assert not result.resumed
    assert len(list(read_documents(result.path))) == 2
    assert (tmp_path / "unit.manifest.json").exists()


def test_token_budget_stops_stream(tmp_path: Path) -> None:
    # budget: 25 tokens ~= 100 chars; each record is 90 chars.
    records = [{"text": "x" * 90} for _ in range(50)]
    result = download_source(_cfg(target_tokens=25), tmp_path, reader=_reader(records))
    assert result.docs == 2  # 90 chars, then 180 >= 100 stops before the third


def test_doc_cap_stops_stream(tmp_path: Path) -> None:
    records = [{"text": f"doc {i} " * 30} for i in range(10)]
    result = download_source(_cfg(doc_cap=3), tmp_path, reader=_reader(records))
    assert result.docs == 3


def test_bad_records_are_skipped_and_counted(tmp_path: Path) -> None:
    records: list[dict[str, Any]] = [{"text": "good " * 30}, {"text": ""}, {"other": "x"}]
    result = download_source(_cfg(), tmp_path, reader=_reader(records))
    assert result.docs == 1
    assert result.skipped == 2


def test_completed_manifest_short_circuits(tmp_path: Path) -> None:
    records = [{"text": "SELECT 1;" * 20}]
    first = download_source(_cfg(), tmp_path, reader=_reader(records))
    second = download_source(_cfg(), tmp_path, reader=_reader([{"text": "changed " * 30}]))
    assert not first.resumed
    assert second.resumed
    assert second.docs == first.docs


def test_synsql_records_are_rendered(tmp_path: Path) -> None:
    records = [{"schema": "CREATE TABLE t (id INT);", "question": "q?", "sql": "SELECT id FROM t;"}]
    cfg = _cfg(name="synsql", kind="synsql", text_field=None)
    result = download_source(cfg, tmp_path, reader=_reader(records))
    doc = next(iter(read_documents(result.path)))
    assert doc.text.startswith("<|schema|>")
    assert doc.source == "synsql"


def test_unknown_structured_source_raises() -> None:
    with pytest.raises(ValueError, match="no registered renderer"):
        record_to_document(_cfg(name="mystery", text_field=None), {"a": "b"}, 0)


def test_schemapile_reader_yields_shaped_records(tmp_path: Path) -> None:
    from sqlpup.data.download import _iter_schemapile

    raw = {
        "a.sql": {
            "INFO": {"LICENSE": "MIT", "PERMISSIVE": True},
            "TABLES": {"t": {"COLUMNS": {"id": {"TYPE": "Int"}}}},
        },
        "b.sql": {"INFO": {}, "TABLES": {}},
    }
    gz = tmp_path / "schemapile.source.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as handle:
        json.dump(raw, handle)

    records = list(_iter_schemapile(gz))
    assert [r["schema_name"] for r in records] == ["a.sql", "b.sql"]
    assert records[0]["TABLES"] == {"t": {"COLUMNS": {"id": {"TYPE": "Int"}}}}


def test_schemapile_records_are_rendered(tmp_path: Path) -> None:
    records = [
        {
            "schema_name": "x.sql",
            "INFO": {"LICENSE": "MIT"},
            "TABLES": {
                "users": {
                    "COLUMNS": {"id": {"TYPE": "Int", "NULLABLE": False, "UNIQUE": True}},
                    "PRIMARY_KEYS": ["id"],
                    "FOREIGN_KEYS": [],
                }
            },
        }
    ]
    cfg = _cfg(name="schemapile", kind="schemapile", text_field=None)
    result = download_source(cfg, tmp_path, reader=_reader(records))
    doc = next(iter(read_documents(result.path)))
    assert doc.text.startswith("<|schema|>")
    assert "CREATE TABLE users (" in doc.text
    assert doc.source == "schemapile"


def test_render_dispatches_on_kind_not_name(tmp_path: Path) -> None:
    # A structured source is chosen by its fetch ``kind``, not its ``name``: the
    # mix plans several synsql/schemapile sources under distinct names (anneal
    # slice, synth schema-Q&A, hard slice), and each must still render.
    synsql_records = [
        {"schema": "CREATE TABLE t (id INT);", "question": "q?", "sql": "SELECT id FROM t;"}
    ]
    synsql_cfg = _cfg(name="synsql_hard", kind="synsql", text_field=None)
    synsql_result = download_source(synsql_cfg, tmp_path, reader=_reader(synsql_records))
    assert (synsql_result.docs, synsql_result.skipped) == (1, 0)
    assert next(iter(read_documents(synsql_result.path))).text.startswith("<|schema|>")

    schemapile_records = [
        {
            "schema_name": "x.sql",
            "INFO": {"LICENSE": "MIT"},
            "TABLES": {
                "users": {
                    "COLUMNS": {"id": {"TYPE": "Int"}},
                    "PRIMARY_KEYS": ["id"],
                    "FOREIGN_KEYS": [],
                }
            },
        }
    ]
    schemapile_cfg = _cfg(name="synth_schema_qa", kind="schemapile", text_field=None)
    schemapile_result = download_source(
        schemapile_cfg, tmp_path, reader=_reader(schemapile_records)
    )
    assert (schemapile_result.docs, schemapile_result.skipped) == (1, 0)
    doc = next(iter(read_documents(schemapile_result.path)))
    assert doc.text.startswith("<|schema|>")
    assert "CREATE TABLE users (" in doc.text


# --- SynSQL streaming fetch adapter -----------------------------------------


def test_iter_json_array_parses_objects_across_chunk_boundaries() -> None:
    from sqlpup.data.download import _iter_json_array

    objs = [
        {"db_id": "café", "sql": "SELECT 1"},  # "é" -> two UTF-8 bytes (C3 A9)
        {"db_id": "b", "sql": "SELECT 2", "nested": {"k": [1, 2, 3]}},
        {"db_id": "c", "sql": "SELECT 3"},
    ]
    # ensure_ascii=False keeps "é" as raw UTF-8 (0xC3 0xA9), not an escaped
    # "\\u00e9", so a chunk boundary can fall between its two bytes.
    raw = json.dumps(objs, ensure_ascii=False).encode("utf-8")
    lead = raw.index(b"\xc3\xa9")  # offset of "é"'s lead byte

    # Cut every 3 bytes to split objects mid-chunk, and force an extra cut
    # between "é"'s two bytes so the incremental UTF-8 decoder must carry the
    # lead byte across a boundary (fixed 3-byte cuts alone leave both bytes in
    # one chunk and never exercise that path).
    bounds = sorted({0, len(raw), lead + 1, *range(3, len(raw), 3)})
    chunks = [raw[a:b] for a, b in pairwise(bounds)]

    # Premise, guarding against silent regression: "é"'s two bytes really do
    # land in different chunks, so the boundary splits the multibyte character.
    owner = [i for i, chunk in enumerate(chunks) for _ in range(len(chunk))]
    assert owner[lead] != owner[lead + 1]

    assert list(_iter_json_array(iter(chunks))) == objs


def test_iter_json_array_handles_empty_array() -> None:
    from sqlpup.data.download import _iter_json_array

    assert list(_iter_json_array(iter([b"[]"]))) == []


def test_iter_json_array_raises_on_truncation_between_elements() -> None:
    from sqlpup.data.download import _iter_json_array

    # A stream cut right after an inter-element comma ('[{"a": 1},' then EOF).
    # Reading this as a clean end would silently mark a truncated 9.36GB
    # download complete, so it must raise.
    raw = json.dumps([{"a": 1}, {"b": 2}]).encode("utf-8")
    truncated = raw[: raw.index(b"}") + 1] + b","
    with pytest.raises(ValueError, match="truncated JSON array"):
        list(_iter_json_array(iter([truncated])))


def test_iter_json_array_raises_on_truncation_after_last_element() -> None:
    from sqlpup.data.download import _iter_json_array

    # A stream cut after the final value but before its closing ']'
    # ('[{"a": 1}' then EOF) -- the other shape of a truncated array.
    raw = json.dumps([{"a": 1}]).encode("utf-8")
    truncated = raw[: raw.rindex(b"]")]
    with pytest.raises(ValueError, match="truncated JSON array"):
        list(_iter_json_array(iter([truncated])))


def test_truncated_stream_capped_early_does_not_raise(tmp_path: Path) -> None:
    from sqlpup.data.download import _iter_json_array

    # The critical nuance: a doc cap tears the streaming generator down at the
    # yield (GeneratorExit) before EOF is ever reached, so the truncation guard
    # must NOT fire on an intentional early stop -- even when the stream really
    # is missing its closing ']'.
    objs = [
        {"db_id": "d", "ddls": ["CREATE TABLE t (id INT)"], "question": "q?", "sql": "SELECT 1"}
        for _ in range(20)
    ]
    raw = json.dumps(objs).encode("utf-8")
    raw = raw[: raw.rindex(b"]")]  # strip closing ']' -> genuinely truncated

    def reader(_: SourceConfig) -> Iterator[Mapping[str, Any]]:
        return _iter_json_array(iter([raw]))

    cfg = _cfg(name="synsql", kind="synsql", text_field=None, doc_cap=2)
    result = download_source(cfg, tmp_path, reader=reader)  # stops at the cap, no raise
    assert result.docs == 2


def test_load_ddl_index_from_tables_json(tmp_path: Path) -> None:
    from sqlpup.data.download import _load_ddl_index

    tables = [
        {"db_id": "school", "ddls": ["CREATE TABLE s (id INT)", "CREATE TABLE u (x INT)"]},
        {"db_id": "shop", "ddls": ["CREATE TABLE item (sku TEXT)"]},
        {"db_id": "blank", "ddls": []},  # no usable DDL -> not indexed
        {"ddls": ["CREATE TABLE orphan (a INT)"]},  # no db_id -> not indexed
    ]
    path = tmp_path / "synsql.tables.json"
    path.write_text(json.dumps(tables), encoding="utf-8")

    index = _load_ddl_index(path)
    assert index["school"] == ["CREATE TABLE s (id INT)", "CREATE TABLE u (x INT)"]
    assert index["shop"] == ["CREATE TABLE item (sku TEXT)"]
    assert "blank" not in index
    assert "orphan" not in index


def test_synsql_join_attaches_ddls_and_skips_missing_db_id(tmp_path: Path) -> None:
    from sqlpup.data.download import _iter_json_array, _join_synsql

    data_objs = [
        {"db_id": "school", "question": "how many?", "sql": "SELECT 1", "cot": "count rows"},
        {"db_id": "unknown_db", "question": "q2", "sql": "SELECT 2"},
        {"db_id": "shop", "question": "q3", "sql": "SELECT 3"},
    ]
    raw = json.dumps(data_objs).encode("utf-8")
    index = {
        "school": ['CREATE TABLE "student" (\n  "id" INTEGER,\n  PRIMARY KEY ("id")\n)'],
        "shop": ['CREATE TABLE "item" (\n  "sku" TEXT\n)'],
    }

    def reader(_: SourceConfig) -> Iterator[Mapping[str, Any]]:
        return _join_synsql(_iter_json_array(iter([raw])), index)

    cfg = _cfg(name="synsql", kind="synsql", text_field=None)
    result = download_source(cfg, tmp_path, reader=reader)
    assert result.docs == 2
    assert result.skipped == 1  # unknown_db has no joined schema -> RenderError -> skipped
    docs = list(read_documents(result.path))
    assert all(d.source == "synsql" for d in docs)
    assert any('CREATE TABLE "student"' in d.text for d in docs)


def test_synsql_stream_stops_early_at_doc_cap(tmp_path: Path) -> None:
    from sqlpup.data.download import _iter_json_array

    objs = [
        {"db_id": "d", "ddls": ["CREATE TABLE t (id INT)"], "question": "q?", "sql": "SELECT 1"}
        for _ in range(20)
    ]
    raw = json.dumps(objs).encode("utf-8")
    all_chunks = [raw[i : i + 16] for i in range(0, len(raw), 16)]
    pulled = 0

    def chunk_gen() -> Iterator[bytes]:
        nonlocal pulled
        for chunk in all_chunks:
            pulled += 1
            yield chunk

    def reader(_: SourceConfig) -> Iterator[Mapping[str, Any]]:
        return _iter_json_array(chunk_gen())

    cfg = _cfg(name="synsql", kind="synsql", text_field=None, doc_cap=3)
    result = download_source(cfg, tmp_path, reader=reader)
    assert result.docs == 3
    # The remainder of the array was never read: streaming stopped at the cap.
    assert pulled < len(all_chunks)


def test_synsql_reader_uses_cached_tables_and_streams_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlpup.data import download as dl

    tables = [{"db_id": "school", "ddls": ['CREATE TABLE "s" (\n  "id" INTEGER\n)']}]
    (tmp_path / "synsql.tables.json").write_text(json.dumps(tables), encoding="utf-8")

    def no_download(url: str, dest: Path) -> None:
        raise AssertionError("tables.json should be served from the cache, not re-downloaded")

    monkeypatch.setattr(dl, "_download_artifact", no_download)

    data_objs = [
        {"db_id": "school", "question": "q", "sql": "SELECT 1"},
        {"db_id": "missing", "question": "q2", "sql": "SELECT 2"},
    ]
    raw = json.dumps(data_objs).encode("utf-8")

    def fake_http_chunks(url: str, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        assert url == "http://example/data.json"
        yield raw

    monkeypatch.setattr(dl, "_iter_http_chunks", fake_http_chunks)

    cfg = _cfg(
        name="synsql",
        kind="synsql",
        text_field=None,
        url="http://example/data.json",
        tables_url="http://example/tables.json",
    )
    records = list(dl.synsql_reader(cfg, tmp_path))
    assert records[0]["db_id"] == "school"
    assert records[0]["ddls"] == ['CREATE TABLE "s" (\n  "id" INTEGER\n)']
    assert "ddls" not in records[1]  # missing db_id -> no schema attached


def test_synsql_resume_short_circuits(tmp_path: Path) -> None:
    records = [
        {"db_id": "school", "ddls": ["CREATE TABLE t (id INT)"], "question": "q", "sql": "S"}
    ]
    cfg = _cfg(name="synsql", kind="synsql", text_field=None)
    first = download_source(cfg, tmp_path, reader=_reader(records))
    second = download_source(cfg, tmp_path, reader=_reader([{"db_id": "other"}]))
    assert not first.resumed
    assert second.resumed
    assert second.docs == first.docs


# --- reader dispatch + cache-hit (offline) ----------------------------------


def test_select_reader_dispatches_on_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlpup.data import download as dl

    # "hf" routes straight to the streaming reader function.
    assert dl._select_reader(_cfg(kind="hf"), tmp_path) is dl.hf_stream_reader

    def fake_schemapile(cfg: SourceConfig, out_dir: Path) -> Iterator[Mapping[str, Any]]:
        yield {"routed": "schemapile", "out_dir": str(out_dir)}

    def fake_synsql(cfg: SourceConfig, out_dir: Path) -> Iterator[Mapping[str, Any]]:
        yield {"routed": "synsql", "out_dir": str(out_dir)}

    monkeypatch.setattr(dl, "schemapile_reader", fake_schemapile)
    monkeypatch.setattr(dl, "synsql_reader", fake_synsql)

    sp_cfg = _cfg(kind="schemapile")
    assert list(dl._select_reader(sp_cfg, tmp_path)(sp_cfg)) == [
        {"routed": "schemapile", "out_dir": str(tmp_path)}
    ]
    sy_cfg = _cfg(kind="synsql")
    assert list(dl._select_reader(sy_cfg, tmp_path)(sy_cfg)) == [
        {"routed": "synsql", "out_dir": str(tmp_path)}
    ]


def test_schemapile_reader_uses_cache_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlpup.data import download as dl

    raw = {"a.sql": {"INFO": {}, "TABLES": {"t": {"COLUMNS": {"id": {"TYPE": "Int"}}}}}}
    cache = tmp_path / "schemapile.source.json.gz"
    with gzip.open(cache, "wt", encoding="utf-8") as handle:
        json.dump(raw, handle)

    def no_download(url: str, dest: Path) -> None:
        raise AssertionError("cache hit must not trigger a network download")

    monkeypatch.setattr(dl, "_download_artifact", no_download)
    cfg = _cfg(name="schemapile", kind="schemapile", text_field=None, url="http://example/x.gz")
    records = list(dl.schemapile_reader(cfg, tmp_path))
    assert [r["schema_name"] for r in records] == ["a.sql"]


# --- hf streaming: builder-config vs directory-style subset selection --------


def test_hf_stream_reader_selects_directory_subset_with_data_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Directory-style subsets (starcoderdata/sql, the-stack-dedup/data/python)
    # are folders in the repo and must be selected with ``data_dir=``; passing
    # them as ``name=`` raises. A source that sets ``hf_data_dir`` therefore
    # must reach ``load_dataset`` with ``data_dir=`` and never with ``name=``.
    import datasets

    from sqlpup.data.download import hf_stream_reader

    calls: list[dict[str, Any]] = []

    def fake_load_dataset(path: str, **kwargs: Any) -> list[Mapping[str, Any]]:
        calls.append({"path": path, **kwargs})
        return [{"content": "SELECT 1;"}]

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    cfg = _cfg(hf_path="bigcode/starcoderdata", hf_name=None, hf_data_dir="sql")
    assert list(hf_stream_reader(cfg)) == [{"content": "SELECT 1;"}]
    assert calls[-1]["path"] == "bigcode/starcoderdata"
    assert calls[-1]["data_dir"] == "sql"
    assert calls[-1]["streaming"] is True
    assert "name" not in calls[-1]


def test_hf_stream_reader_selects_builder_config_with_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Builder-config subsets (FineWeb-Edu's sample-10BT) keep the ``name=`` path
    # and must not sprout a ``data_dir=`` that would break them.
    import datasets

    from sqlpup.data.download import hf_stream_reader

    calls: list[dict[str, Any]] = []

    def fake_load_dataset(path: str, **kwargs: Any) -> list[Mapping[str, Any]]:
        calls.append({"path": path, **kwargs})
        return [{"text": "hello"}]

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    cfg = _cfg(hf_path="HuggingFaceFW/fineweb-edu", hf_name="sample-10BT")
    assert list(hf_stream_reader(cfg)) == [{"text": "hello"}]
    assert calls[-1]["name"] == "sample-10BT"
    assert "data_dir" not in calls[-1]
