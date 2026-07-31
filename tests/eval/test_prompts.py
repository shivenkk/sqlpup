"""Prompt rendering -- the byte-exact SFT/eval contract, pinned by a golden file."""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path

import pytest

from sqlpup.eval.prompts import BIRD_DDL_V1, schema_ddl

GOLDEN = Path(__file__).parent / "fixtures" / "prompts" / "bird_ddl_v1.txt"


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE t (id INTEGER, name TEXT);
        CREATE TABLE nums (n INTEGER);
        """
    )
    con.commit()
    con.close()
    return path


def test_schema_ddl_is_verbatim_name_ordered_and_terminated(fixture_db: Path) -> None:
    ddl = schema_ddl(fixture_db)
    assert ddl == "CREATE TABLE nums (n INTEGER);\n\nCREATE TABLE t (id INTEGER, name TEXT);"


def test_schema_ddl_skips_sqlite_internal_tables(tmp_path: Path) -> None:
    path = tmp_path / "auto.sqlite"
    con = sqlite3.connect(path)
    # An AUTOINCREMENT table materialises the internal sqlite_sequence table.
    con.execute("CREATE TABLE a (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    con.commit()
    con.close()
    assert "sqlite_sequence" not in schema_ddl(path)


def test_render_matches_golden_file(fixture_db: Path) -> None:
    prompt = BIRD_DDL_V1.render(
        schema_ddl(fixture_db),
        question="How many rows does t have?",
        evidence="rows refers to COUNT(*)",
    )
    assert prompt == GOLDEN.read_text(encoding="utf-8")


def test_render_omits_evidence_block_when_empty(fixture_db: Path) -> None:
    prompt = BIRD_DDL_V1.render(schema_ddl(fixture_db), question="q", evidence="  ")
    assert "External knowledge" not in prompt
    assert "-- Question: q\n" in prompt


def test_render_repair_appends_failure_and_new_cue() -> None:
    repaired = BIRD_DDL_V1.render_repair("PROMPT\n-- SQL:\n", "SELEC 1", "OperationalError: near")
    assert repaired.startswith("PROMPT\n-- SQL:\n")
    assert "SELEC 1" in repaired
    assert "OperationalError: near" in repaired
    assert repaired.endswith("-- Corrected SQL:\n")


def test_spec_id_is_stable_and_content_addressed() -> None:
    assert BIRD_DDL_V1.spec_id.startswith("bird-ddl-v1@")
    changed = dataclasses.replace(BIRD_DDL_V1, template=BIRD_DDL_V1.template + " ")
    assert changed.spec_id != BIRD_DDL_V1.spec_id


def test_prompt_spec_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        BIRD_DDL_V1.name = "other"  # type: ignore[misc]


def test_selectcue_variant_ends_in_statement_mode(fixture_db: Path) -> None:
    from sqlpup.eval.prompts import BIRD_DDL_V1_SELECTCUE

    prompt = BIRD_DDL_V1_SELECTCUE.render(schema_ddl(fixture_db), question="q", evidence="")
    assert prompt.endswith("-- SQL:\nSELECT")
    assert BIRD_DDL_V1_SELECTCUE.completion_prefix == "SELECT"
    repaired = BIRD_DDL_V1_SELECTCUE.render_repair(prompt, "SELECT 1", "err")
    assert repaired.endswith("-- Corrected SQL:\nSELECT")


def test_spec_registry_maps_names_to_specs() -> None:
    from sqlpup.eval.prompts import BIRD_DDL_V1_SELECTCUE, SPECS

    assert SPECS["bird-ddl-v1"] is BIRD_DDL_V1
    assert SPECS["bird-ddl-v1-selectcue"] is BIRD_DDL_V1_SELECTCUE
    assert BIRD_DDL_V1.spec_id != BIRD_DDL_V1_SELECTCUE.spec_id
