import json
from pathlib import Path
from typing import Any

import pytest

from sqlpup.data.render import RenderError, render_schemapile, render_synsql

FIXTURES = Path(__file__).parent / "fixtures"

FULL_RECORD = {
    "db_id": "school",
    "schema": "CREATE TABLE t (id INT);",
    "question": "How many rows?",
    "cot": "Count rows in t.",
    "sql": "SELECT COUNT(*) FROM t;",
}


def _load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_full_synsql_record_renders_exactly() -> None:
    doc = render_synsql(FULL_RECORD, doc_id="synsql-0")
    assert doc.text == (
        "<|schema|>\nCREATE TABLE t (id INT);\n"
        "<|question|>\nHow many rows?\n"
        "<|think|>\nCount rows in t.\n"
        "<|sql|>\nSELECT COUNT(*) FROM t;"
    )
    assert doc.source == "synsql"
    assert doc.meta == {"db_id": "school"}


def test_missing_cot_omits_think_block() -> None:
    record = {k: v for k, v in FULL_RECORD.items() if k != "cot"}
    doc = render_synsql(record, doc_id="synsql-1")
    assert "<|think|>" not in doc.text
    assert "<|sql|>" in doc.text


def test_field_aliases_are_accepted() -> None:
    record = {"ddl": "CREATE TABLE x (a INT);", "question": "q?", "SQL": "SELECT a FROM x;"}
    doc = render_synsql(record, doc_id="synsql-2")
    assert doc.text.startswith("<|schema|>\nCREATE TABLE x (a INT);")


def test_missing_sql_raises_with_available_keys() -> None:
    record = {"schema": "CREATE TABLE t (id INT);", "question": "q?"}
    with pytest.raises(RenderError, match="available keys: question, schema"):
        render_synsql(record, doc_id="synsql-3")


def test_synsql_real_records_render_with_joined_schema() -> None:
    records = _load_fixture("synsql_sample.json")
    assert records, "fixture must contain real records"
    for i, rec in enumerate(records):
        doc = render_synsql(rec, doc_id=f"synsql-{i}")
        assert doc.source == "synsql"
        assert doc.meta == {"db_id": rec["db_id"]}
        # the schema block is reconstructed from the real ``ddls`` list
        assert doc.text.startswith(f"<|schema|>\n{rec['ddls'][0]}")
        for ddl in rec["ddls"]:
            assert ddl in doc.text
        assert f"<|question|>\n{rec['question']}" in doc.text
        assert f"<|think|>\n{rec['cot']}" in doc.text
        assert doc.text.endswith(f"<|sql|>\n{rec['sql']}")
        text = doc.text
        assert (
            text.index("<|schema|>")
            < text.index("<|question|>")
            < text.index("<|think|>")
            < text.index("<|sql|>")
        )


def test_schemapile_real_records_render_as_create_table() -> None:
    fixtures = _load_fixture("schemapile_sample.json")
    assert fixtures, "fixture must contain real schemas"
    for name, value in fixtures.items():
        doc = render_schemapile({"schema_name": name, **value}, doc_id=f"sp-{name}")
        assert doc.source == "schemapile"
        assert doc.meta == {"schema_name": name}
        assert doc.text.startswith("<|schema|>\n")
        assert not doc.text.endswith("<|end|>")  # ShardWriter owns EOS, not the renderer
        for tname, table in value["TABLES"].items():
            assert f"CREATE TABLE {tname} (" in doc.text
            for column in table["COLUMNS"]:
                assert column in doc.text


def test_schemapile_renders_keys_constraints_and_nullability() -> None:
    fixtures = _load_fixture("schemapile_sample.json")
    fk_name = next(
        name
        for name, value in fixtures.items()
        for table in value["TABLES"].values()
        if table.get("FOREIGN_KEYS")
    )
    doc = render_schemapile({"schema_name": fk_name, **fixtures[fk_name]}, doc_id="sp-fk")
    assert "PRIMARY KEY (" in doc.text
    assert "FOREIGN KEY (" in doc.text
    assert "REFERENCES" in doc.text
    assert "NOT NULL" in doc.text  # a NULLABLE=false column


def test_schemapile_requires_tables() -> None:
    with pytest.raises(RenderError, match="no tables"):
        render_schemapile({"schema_name": "empty.sql", "TABLES": {}}, doc_id="sp-empty")


def test_renderers_emit_no_trailing_end_token() -> None:
    # ShardWriter appends the EOS id after every document; if a renderer also
    # emitted a trailing <|end|>, tokenized shards would carry a doubled
    # separator (<|end|><|end|>). Renderers emit section tokens only.
    synsql = render_synsql(FULL_RECORD, doc_id="synsql-eos")
    assert "<|end|>" not in synsql.text
    assert synsql.text.endswith("SELECT COUNT(*) FROM t;")

    schemapile = _load_fixture("schemapile_sample.json")
    name, value = next(iter(schemapile.items()))
    sp_doc = render_schemapile({"schema_name": name, **value}, doc_id="sp-eos")
    assert "<|end|>" not in sp_doc.text
