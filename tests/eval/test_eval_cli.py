"""End-to-end ``sqlpup eval`` on a synthetic archive + predictions file (offline)."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from sqlpup.cli import main
from sqlpup.eval.dataset import BIRD_DEV_URL


def _cli_db_bytes() -> bytes:
    path = Path("/tmp/_sqlpup_cli_db.sqlite")
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (n INTEGER)")
    con.executemany("INSERT INTO t (n) VALUES (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()
    data = path.read_bytes()
    path.unlink(missing_ok=True)
    return data


def _write_dev_archive(eval_dir: Path) -> None:
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dev_databases/cli_db/cli_db.sqlite", _cli_db_bytes())
    examples = [
        {
            "question_id": 0,
            "db_id": "cli_db",
            "question": "all n",
            "evidence": "",
            "SQL": "SELECT n FROM t ORDER BY n",
            "difficulty": "simple",
        },
        {
            "question_id": 1,
            "db_id": "cli_db",
            "question": "n=1",
            "evidence": "",
            "SQL": "SELECT n FROM t WHERE n = 1",
            "difficulty": "moderate",
        },
    ]
    outer_buf = io.BytesIO()
    with zipfile.ZipFile(outer_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dev_20240627/dev.json", json.dumps(examples))
        zf.writestr("dev_20240627/dev_databases.zip", inner_buf.getvalue())
    digest = hashlib.sha256(BIRD_DEV_URL.encode("utf-8")).hexdigest()[:16]
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{digest}.zip").write_bytes(outer_buf.getvalue())


def test_eval_command_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    eval_dir = tmp_path / "eval"
    _write_dev_archive(eval_dir)

    predictions = tmp_path / "preds.jsonl"
    predictions.write_text(
        '{"index": 0, "db_id": "cli_db", "predicted_sql": "SELECT n FROM t ORDER BY n"}\n'
        # a deliberately wrong second prediction -> EX 0.5
        '{"index": 1, "db_id": "cli_db", "predicted_sql": "SELECT n FROM t"}\n',
        encoding="utf-8",
    )
    out_path = tmp_path / "artifacts" / "eval_detail.json"

    code = main(
        [
            "eval",
            "score",
            "--predictions",
            str(predictions),
            "--subset",
            "dev",
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(out_path),
        ]
    )
    assert code == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["subset"] == "dev"
    assert summary["total"] == 2
    assert summary["correct"] == 1
    assert summary["ex"] == 0.5
    assert summary["by_difficulty"]["simple"]["ex"] == 1.0
    assert summary["by_difficulty"]["moderate"]["ex"] == 0.0

    # the detailed per-example artifact is written to --out
    detail = json.loads(out_path.read_text(encoding="utf-8"))
    assert detail["summary"] == summary
    assert [e["index"] for e in detail["examples"]] == [0, 1]
    assert detail["examples"][0]["match"] is True
    assert detail["examples"][1]["match"] is False


def test_eval_command_defaults_timeout_and_row_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No --out, default timeout/row-limit: still emits a summary and exits 0.
    eval_dir = tmp_path / "eval"
    _write_dev_archive(eval_dir)
    predictions = tmp_path / "preds.jsonl"
    predictions.write_text(
        '{"index": 0, "predicted_sql": "SELECT n FROM t ORDER BY n"}\n'
        '{"index": 1, "predicted_sql": "SELECT n FROM t WHERE n = 1"}\n',
        encoding="utf-8",
    )
    assert (
        main(["eval", "score", "--predictions", str(predictions), "--eval-dir", str(eval_dir)]) == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["ex"] == 1.0  # both predictions equal gold


def _write_gold_check_eval_dir(tmp_path: Path, gold_sql: str) -> Path:
    """A mini-dev eval dir with one database and one example using ``gold_sql``.

    Databases are laid out directly (no archive), so ``ensure_databases`` finds
    everything present and the check stays fully offline.
    """
    eval_dir = tmp_path / "eval"
    db_dir = eval_dir / "dev_databases" / "toy"
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / "toy.sqlite")
    con.executescript("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1);")
    con.commit()
    con.close()
    (eval_dir / "mini_dev_sqlite.json").write_text(
        json.dumps(
            [
                {
                    "question_id": 0,
                    "db_id": "toy",
                    "question": "q",
                    "evidence": "",
                    "SQL": gold_sql,
                    "difficulty": "simple",
                }
            ]
        ),
        encoding="utf-8",
    )
    return eval_dir


def test_gold_check_passes_on_healthy_gold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = _write_gold_check_eval_dir(tmp_path, "SELECT id FROM t")
    rc = main(["eval", "gold-check", "--eval-dir", str(eval_dir), "--subset", "mini-dev"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 1
    assert payload["passed"] == 1
    assert payload["failures"] == []


def test_gold_check_fails_loudly_on_broken_gold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = _write_gold_check_eval_dir(tmp_path, "SELEC id FROM t")
    rc = main(["eval", "gold-check", "--eval-dir", str(eval_dir), "--subset", "mini-dev"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] == 0
    assert payload["failures"][0]["db_id"] == "toy"
    assert payload["failures"][0]["category"] == "gold_error"


def test_gold_check_can_archive_its_certificate_to_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every other eval subcommand takes ``--out``; gold-check did not, so the
    side-eval boxes' archive step failed with ``unrecognized arguments`` on every
    run. Harmless (it sits in a pipeline) but it prints a hard error into logs
    that BIRD's organisers will read when they run this code themselves.
    """
    eval_dir = _write_gold_check_eval_dir(tmp_path, "SELECT id FROM t")
    out = tmp_path / "gold.json"
    rc = main(
        [
            "eval",
            "gold-check",
            "--eval-dir",
            str(eval_dir),
            "--subset",
            "mini-dev",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    archived = json.loads(out.read_text(encoding="utf-8"))
    assert archived["total"] == 1 and archived["passed"] == 1
    # stdout stays machine-readable so existing pipelines keep working
    assert json.loads(capsys.readouterr().out)["passed"] == 1


def test_generate_accepts_a_sampling_seed(tmp_path: Path) -> None:
    """Self-consistency seeds each draw as manual_seed(seed + draw), so a run is
    fully deterministic. Without a --seed flag every voting run reused seed 0,
    which is why k=7 and k=8 shared their first seven candidates exactly. Error
    bars on the headline number require varying it."""
    from sqlpup.cli import build_parser

    args = build_parser().parse_args(
        ["eval", "generate", "--model-dir", "m", "--out", "o.json", "--seed", "3"]
    )
    assert args.seed == 3
    default = build_parser().parse_args(["eval", "generate", "--model-dir", "m", "--out", "o.json"])
    assert default.seed == 0  # every run so far
