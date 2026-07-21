"""Render structured NL-to-SQL records into pretraining documents.

The pretraining corpus teaches the schema/question/reasoning/SQL layout with
the same special tokens the tokenizer reserves and the fine-tuning prompt
format reuses, so nothing about the format is novel to the model at SFT time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlpup.io import Document

SCHEMA_TOKEN = "<|schema|>"
QUESTION_TOKEN = "<|question|>"
THINK_TOKEN = "<|think|>"
SQL_TOKEN = "<|sql|>"
# The document/EOS separator (<|end|>) is owned by ShardWriter, which appends it
# after every tokenized document; renderers emit section tokens only so shards
# do not carry a doubled separator.

_SCHEMA_KEYS = ("schema", "ddl", "create_statements")
_DDL_LIST_KEY = "ddls"
_QUESTION_KEYS = ("question",)
_SQL_KEYS = ("sql", "SQL", "query")
_COT_KEYS = ("cot", "chain_of_thought", "reasoning")


class RenderError(ValueError):
    """Raised when a record is missing a field required by its template."""


def _first(record: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _require(record: Mapping[str, Any], keys: tuple[str, ...], what: str) -> str:
    value = _first(record, keys)
    if value is None:
        available = ", ".join(sorted(record.keys()))
        raise RenderError(f"record missing {what} (tried {keys}); available keys: {available}")
    return value


def _synsql_schema(record: Mapping[str, Any]) -> str | None:
    """Schema DDL from SynSQL's ``ddls`` list, or an inline schema string.

    Real ``seeklhy/SynSQL-2.5M`` records carry the schema in a separate
    ``tables.json`` as a ``ddls`` list of ``CREATE TABLE`` statements (joined
    onto the record by ``db_id``); older/inline shapes use a schema string.
    """
    ddls = record.get(_DDL_LIST_KEY)
    if isinstance(ddls, list):
        parts = [d.strip() for d in ddls if isinstance(d, str) and d.strip()]
        if parts:
            return "\n\n".join(parts)
    return _first(record, _SCHEMA_KEYS)


def render_synsql(record: Mapping[str, Any], doc_id: str) -> Document:
    """Render a SynSQL-style record: schema + question (+ reasoning) + SQL."""
    schema = _synsql_schema(record)
    if schema is None:
        available = ", ".join(sorted(record.keys()))
        raise RenderError(
            f"record missing schema DDL (tried {_DDL_LIST_KEY!r} list and {_SCHEMA_KEYS}); "
            f"available keys: {available}"
        )
    question = _require(record, _QUESTION_KEYS, "question")
    sql = _require(record, _SQL_KEYS, "gold SQL")
    cot = _first(record, _COT_KEYS)

    parts = [f"{SCHEMA_TOKEN}\n{schema}", f"{QUESTION_TOKEN}\n{question}"]
    if cot is not None:
        parts.append(f"{THINK_TOKEN}\n{cot}")
    parts.append(f"{SQL_TOKEN}\n{sql}")
    text = "\n".join(parts)

    meta = {"db_id": str(record["db_id"])} if "db_id" in record else {}
    return Document(id=doc_id, text=text, source="synsql", meta=meta)


def _schemapile_ddl(tables: Mapping[str, Any]) -> str:
    """Reconstruct ``CREATE TABLE`` statements from SchemaPile's structured tables.

    Real SchemaPile records describe each table as ``{COLUMNS, PRIMARY_KEYS,
    FOREIGN_KEYS, ...}`` rather than as DDL text, so the schema is rebuilt here
    to match the DDL layout the model sees from the other schema sources.
    """
    statements: list[str] = []
    for table_name, table in tables.items():
        if not isinstance(table, Mapping):
            continue
        columns = table.get("COLUMNS")
        if not isinstance(columns, Mapping):
            continue
        lines: list[str] = []
        for column_name, raw_spec in columns.items():
            spec = raw_spec if isinstance(raw_spec, Mapping) else {}
            parts = [str(column_name), str(spec.get("TYPE", ""))]
            if spec.get("NULLABLE") is False:
                parts.append("NOT NULL")
            if spec.get("UNIQUE") is True:
                parts.append("UNIQUE")
            default = spec.get("DEFAULT")
            if default is not None:
                parts.append(f"DEFAULT {default}")
            lines.append("  " + " ".join(part for part in parts if part))
        primary = [str(col) for col in table.get("PRIMARY_KEYS") or []]
        if primary:
            lines.append(f"  PRIMARY KEY ({', '.join(primary)})")
        for fk in table.get("FOREIGN_KEYS") or []:
            if not isinstance(fk, Mapping):
                continue
            cols = ", ".join(str(col) for col in fk.get("COLUMNS") or [])
            ref_table = str(fk.get("FOREIGN_TABLE", ""))
            ref_cols = ", ".join(str(col) for col in fk.get("REFERRED_COLUMNS") or [])
            if cols and ref_table:
                lines.append(f"  FOREIGN KEY ({cols}) REFERENCES {ref_table} ({ref_cols})")
        if lines:
            statements.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(lines) + "\n)")
    return "\n\n".join(statements)


def render_schemapile(record: Mapping[str, Any], doc_id: str) -> Document:
    """Render a SchemaPile record: structured tables reconstructed as schema DDL."""
    tables = record.get("TABLES")
    if not isinstance(tables, Mapping) or not tables:
        available = ", ".join(sorted(record.keys()))
        raise RenderError(f"record has no tables; available keys: {available}")
    schema = _schemapile_ddl(tables)
    if not schema:
        raise RenderError("record has no renderable table columns")
    text = f"{SCHEMA_TOKEN}\n{schema}"
    meta = {"schema_name": str(record["schema_name"])} if "schema_name" in record else {}
    return Document(id=doc_id, text=text, source="schemapile", meta=meta)
