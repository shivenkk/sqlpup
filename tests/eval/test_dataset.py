"""BIRD dev / Mini-Dev data access: example loading + idempotent DB extraction.

Offline only: a synthetic outer archive (dev.json + a nested dev_databases.zip
holding a tiny real SQLite file) stands in for the 346MB BIRD download, so the
zip-streaming, extraction, and idempotency paths are exercised without the real
databases or the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from sqlpup.eval.dataset import (
    BIRD_DEV_URL,
    MINI_DEV_MEMBER,
    MINI_DEV_URL,
    BirdExample,
    ensure_databases,
    load_examples,
    resolve_db_path,
)

DB_IDS = ("mini_holdings", "toy_shop")


def _tiny_sqlite_bytes(db_id: str) -> bytes:
    """A minimal but real SQLite file so resolve/extract can be validated."""
    path = Path(f"/tmp/_sqlpup_fixture_{db_id}.sqlite")
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE marker (db TEXT)")
    con.execute("INSERT INTO marker (db) VALUES (?)", (db_id,))
    con.commit()
    con.close()
    data = path.read_bytes()
    path.unlink(missing_ok=True)
    return data


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def _dev_examples() -> list[dict[str, object]]:
    return [
        {
            "question_id": 0,
            "db_id": "mini_holdings",
            "question": "How many holdings are there?",
            "evidence": "",
            "SQL": "SELECT COUNT(*) FROM marker",
            "difficulty": "simple",
        },
        {
            "question_id": 1,
            "db_id": "toy_shop",
            "question": "List the toys.",
            "evidence": "a hint",
            "SQL": "SELECT db FROM marker",
            "difficulty": "moderate",
        },
    ]


def _write_dev_archive(eval_dir: Path) -> None:
    """Place a synthetic BIRD dev archive at the URL-derived cache path."""
    inner = _zip_bytes(
        {f"dev_databases/{db_id}/{db_id}.sqlite": _tiny_sqlite_bytes(db_id) for db_id in DB_IDS}
        | {"dev_databases/mini_holdings/database_description/marker.csv": b"db\n"}
    )
    outer = _zip_bytes(
        {
            "dev_20240627/dev.json": json.dumps(_dev_examples()).encode("utf-8"),
            "dev_20240627/dev_databases.zip": inner,
        }
    )
    digest = hashlib.sha256(BIRD_DEV_URL.encode("utf-8")).hexdigest()[:16]
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{digest}.zip").write_bytes(outer)


# --- dev example loading -----------------------------------------------------


def test_load_dev_examples_from_cached_archive(tmp_path: Path) -> None:
    _write_dev_archive(tmp_path)
    examples = load_examples(tmp_path, "dev")
    assert [e.index for e in examples] == [0, 1]
    assert isinstance(examples[0], BirdExample)
    assert examples[0].db_id == "mini_holdings"
    assert examples[0].gold_sql == "SELECT COUNT(*) FROM marker"
    assert examples[0].difficulty == "simple"
    assert examples[0].question_id == "0"  # coerced to str
    assert examples[1].evidence == "a hint"


# --- idempotent database extraction ------------------------------------------


def test_ensure_databases_extracts_from_nested_zip(tmp_path: Path) -> None:
    _write_dev_archive(tmp_path)
    db_root = ensure_databases(tmp_path, DB_IDS)
    for db_id in DB_IDS:
        path = db_root / db_id / f"{db_id}.sqlite"
        assert path.exists()
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        assert con.execute("SELECT db FROM marker").fetchone() == (db_id,)
        con.close()


def test_ensure_databases_is_idempotent_and_does_not_reextract(tmp_path: Path) -> None:
    _write_dev_archive(tmp_path)
    db_root = ensure_databases(tmp_path, DB_IDS)
    # A sentinel dropped beside the extracted DB must survive a second call: a
    # re-extraction (or a wipe) would remove it.
    sentinel = db_root / "mini_holdings" / "SENTINEL"
    sentinel.write_text("keep me", encoding="utf-8")
    again = ensure_databases(tmp_path, DB_IDS)
    assert again == db_root
    assert sentinel.exists()


def test_ensure_databases_skips_when_already_present_without_archive(tmp_path: Path) -> None:
    # DBs already on disk, no archive at all -> must not need to extract/download.
    for db_id in DB_IDS:
        target = resolve_db_path(tmp_path, db_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_tiny_sqlite_bytes(db_id))
    db_root = ensure_databases(tmp_path, DB_IDS)
    assert db_root.exists()


def test_resolve_db_path_shape(tmp_path: Path) -> None:
    path = resolve_db_path(tmp_path, "toy_shop")
    assert path == tmp_path / "dev_databases" / "toy_shop" / "toy_shop.sqlite"


# --- mini-dev loading --------------------------------------------------------


def test_load_mini_dev_keys_by_index_not_question_id(tmp_path: Path) -> None:
    # Mini-Dev's question_id is NOT unique; loading must key by position so two
    # rows sharing a question_id stay distinct examples.
    rows = [
        {
            "question_id": 5,
            "db_id": "toy_shop",
            "question": "q a",
            "evidence": "",
            "SQL": "SELECT 1",
            "difficulty": "simple",
        },
        {
            "question_id": 5,
            "db_id": "toy_shop",
            "question": "q b",
            "evidence": "",
            "SQL": "SELECT 2",
            "difficulty": "challenging",
        },
    ]
    (tmp_path / "mini_dev_sqlite.json").write_text(json.dumps(rows), encoding="utf-8")
    examples = load_examples(tmp_path, "mini-dev")
    assert [e.index for e in examples] == [0, 1]
    assert examples[0].question_id == examples[1].question_id == "5"
    assert examples[0].gold_sql == "SELECT 1"
    assert examples[1].gold_sql == "SELECT 2"


def test_load_mini_dev_missing_file_names_url_and_member(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        load_examples(tmp_path, "mini-dev")
    message = str(excinfo.value)
    assert MINI_DEV_URL in message
    assert MINI_DEV_MEMBER in message


def test_unknown_subset_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="subset"):
        load_examples(tmp_path, "train")


def test_load_bird_file_reads_a_raw_train_split(tmp_path: Path) -> None:
    """GRPO trains on BIRD-train, which ships as a plain JSON file rather than
    through the dev archive machinery."""
    import json as _json

    from sqlpup.eval.dataset import load_bird_file

    path = tmp_path / "train.json"
    path.write_text(
        _json.dumps(
            [
                {"db_id": "a", "question": "q1", "SQL": "SELECT 1", "evidence": "e"},
                {"db_id": "b", "question": "q2", "SQL": "SELECT 2"},
            ]
        )
    )
    rows = load_bird_file(path)
    assert [r.db_id for r in rows] == ["a", "b"]
    assert rows[0].gold_sql == "SELECT 1"
    assert rows[0].evidence == "e"
    assert rows[1].evidence == ""
    assert [r.index for r in rows] == [0, 1]


def test_load_bird_file_accepts_the_lowercase_sql_key(tmp_path: Path) -> None:
    """BIRD-train says ``SQL``; SynSQL-style rows say ``sql``. Accepting both
    keeps one loader for every source we train on."""
    import json as _json

    from sqlpup.eval.dataset import load_bird_file

    path = tmp_path / "rows.json"
    path.write_text(_json.dumps([{"db_id": "a", "question": "q", "sql": "SELECT 1"}]))
    assert load_bird_file(path)[0].gold_sql == "SELECT 1"


def test_load_bird_file_rejects_a_row_with_no_query(tmp_path: Path) -> None:
    import json as _json

    from sqlpup.eval.dataset import load_bird_file

    path = tmp_path / "bad.json"
    path.write_text(_json.dumps([{"db_id": "a", "question": "q"}]))
    with pytest.raises(KeyError):
        load_bird_file(path)
