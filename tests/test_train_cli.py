"""End-to-end tests for the ``sqlpup train`` CLI (torch-backed).

Each test drives the real :func:`sqlpup.cli.main` entrypoint over a tiny model
and synthetic learnable shards in ``tmp_path``, so a full train -> checkpoint ->
resume cycle runs in a fraction of a second. The tests skip when torch is
absent; the missing-torch friendly-error test lives in ``test_cli.py`` so it
runs regardless of the optional extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sqlpup.cli import main
from sqlpup.shard import ShardWriter

VOCAB_SIZE = 64
EOS_ID = VOCAB_SIZE - 1
PATTERN = list(range(8))


def _write_shards(out_dir: Path, *, documents: int, repeats: int) -> None:
    writer = ShardWriter(out_dir, eos_id=EOS_ID, shard_size_tokens=10_000_000)
    for _ in range(documents):
        writer.add(PATTERN * repeats)
    writer.close()


def _write_model_config(path: Path) -> None:
    path.write_text(
        "d_model: 16\n"
        "n_layers: 1\n"
        "n_heads: 2\n"
        "n_kv_heads: 1\n"
        "d_ff: 32\n"
        "vocab_size: 64\n"
        "max_seq_len: 16\n",
        encoding="utf-8",
    )


def _write_train_config(
    path: Path,
    *,
    model_config: Path,
    train_index: Path,
    eval_index: Path,
    out_dir: Path,
    total_steps: int,
    decay_start_step: int,
) -> None:
    path.write_text(
        f"model_config: {model_config}\n"
        f"shard_index: {train_index}\n"
        f"eval_shard_index: {eval_index}\n"
        f"out_dir: {out_dir}\n"
        "seq_len: 16\n"
        "micro_batch_size: 2\n"
        "grad_accum_steps: 1\n"
        "max_lr: 1.0e-2\n"
        "warmup_steps: 1\n"
        f"decay_start_step: {decay_start_step}\n"
        f"total_steps: {total_steps}\n"
        "eval_interval_steps: 2\n"
        "eval_batches: 1\n"
        "checkpoint_interval_steps: 2\n"
        "log_interval_steps: 1\n"
        "seed: 0\n",
        encoding="utf-8",
    )


@pytest.fixture()
def env(tmp_path: Path) -> dict[str, Path]:
    """A tiny model config, train/eval shards, and an out_dir under ``tmp_path``."""
    pytest.importorskip("torch")
    model_cfg = tmp_path / "model.yaml"
    _write_model_config(model_cfg)
    train_dir = tmp_path / "shards" / "train"
    eval_dir = tmp_path / "shards" / "eval"
    _write_shards(train_dir, documents=8, repeats=4)
    _write_shards(eval_dir, documents=4, repeats=4)
    return {
        "tmp": tmp_path,
        "model_cfg": model_cfg,
        "train_index": train_dir / "index.json",
        "eval_index": eval_dir / "index.json",
        "out_dir": tmp_path / "ckpt",
    }


def _stats(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines, "train CLI emitted no stats line on stdout"
    parsed: dict[str, object] = json.loads(lines[-1])
    return parsed


def _config_for(
    env: dict[str, Path], name: str, *, total_steps: int, decay_start_step: int
) -> Path:
    cfg = env["tmp"] / name
    _write_train_config(
        cfg,
        model_config=env["model_cfg"],
        train_index=env["train_index"],
        eval_index=env["eval_index"],
        out_dir=env["out_dir"],
        total_steps=total_steps,
        decay_start_step=decay_start_step,
    )
    return cfg


def test_train_fresh_run_writes_checkpoint_and_stats(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _config_for(env, "train.yaml", total_steps=4, decay_start_step=3)
    assert main(["train", "--config", str(cfg), "--device", "cpu"]) == 0
    stats = _stats(capsys)
    assert stats["resumed"] is False
    assert stats["start_step"] == 0
    assert stats["step"] == 4
    assert isinstance(stats["final_loss"], float)
    assert "eval_loss" in stats and "eval_ppl" in stats
    assert stats["tokens_seen"] == 2 * 1 * 16 * 4  # mb * accum * seq_len * steps
    assert (env["out_dir"] / "latest.json").exists()
    assert list(env["out_dir"].glob("step_*.pt"))
    assert stats["checkpoint"] == str(env["out_dir"] / "step_4.pt")


def test_resume_auto_without_checkpoint_starts_fresh(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _config_for(env, "train.yaml", total_steps=4, decay_start_step=3)
    # out_dir has no latest.json yet: --resume auto must fall back to a fresh start.
    assert main(["train", "--config", str(cfg), "--device", "cpu", "--resume", "auto"]) == 0
    stats = _stats(capsys)
    assert stats["resumed"] is False
    assert stats["start_step"] == 0
    assert stats["step"] == 4


def test_resume_auto_with_checkpoint_continues(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    fresh = _config_for(env, "fresh.yaml", total_steps=4, decay_start_step=3)
    assert main(["train", "--config", str(fresh), "--device", "cpu"]) == 0
    capsys.readouterr()  # discard the fresh-run stats line

    extended = _config_for(env, "extended.yaml", total_steps=8, decay_start_step=6)
    assert main(["train", "--config", str(extended), "--device", "cpu", "--resume", "auto"]) == 0
    stats = _stats(capsys)
    assert stats["resumed"] is True
    assert stats["start_step"] == 4  # continued from the checkpoint, not 0
    assert stats["step"] == 8


def test_resume_explicit_path_continues(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    fresh = _config_for(env, "fresh.yaml", total_steps=4, decay_start_step=3)
    assert main(["train", "--config", str(fresh), "--device", "cpu"]) == 0
    capsys.readouterr()
    checkpoint = env["out_dir"] / "step_4.pt"
    assert checkpoint.exists()

    extended = _config_for(env, "extended.yaml", total_steps=8, decay_start_step=6)
    assert (
        main(["train", "--config", str(extended), "--device", "cpu", "--resume", str(checkpoint)])
        == 0
    )
    stats = _stats(capsys)
    assert stats["resumed"] is True
    assert stats["start_step"] == 4
    assert stats["step"] == 8


def test_resume_explicit_missing_path_errors(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _config_for(env, "train.yaml", total_steps=4, decay_start_step=3)
    code = main(["train", "--config", str(cfg), "--device", "cpu", "--resume", "nope.pt"])
    assert code != 0
    assert "not found" in capsys.readouterr().err


def test_out_dir_override_redirects_checkpoints(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _config_for(env, "train.yaml", total_steps=4, decay_start_step=3)
    override = env["tmp"] / "elsewhere"
    assert main(["train", "--config", str(cfg), "--device", "cpu", "--out-dir", str(override)]) == 0
    stats = _stats(capsys)
    assert (override / "latest.json").exists()
    assert not env["out_dir"].exists()  # the config's out_dir was overridden
    assert stats["checkpoint"] == str(override / "step_4.pt")


def test_bare_relaunch_over_existing_run_refuses_without_destroying(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _config_for(env, "train.yaml", total_steps=4, decay_start_step=3)
    assert main(["train", "--config", str(cfg), "--device", "cpu"]) == 0
    capsys.readouterr()  # discard the first run's stats

    latest = env["out_dir"] / "latest.json"
    latest_before = latest.read_text(encoding="utf-8")
    files_before = sorted(p.name for p in env["out_dir"].glob("step_*.pt"))

    # A bare relaunch (no --resume, no --force-fresh) must refuse rather than
    # silently overwrite/prune the existing checkpoints.
    code = main(["train", "--config", str(cfg), "--device", "cpu"])
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""  # no stats line emitted
    assert str(env["out_dir"]) in captured.err  # names the offending out_dir
    assert "step 4" in captured.err  # names the existing checkpoint step
    assert "--resume auto" in captured.err and "--force-fresh" in captured.err  # guidance
    assert "Traceback" not in captured.err  # friendly, not a stack trace
    # nothing on disk was touched
    assert latest.read_text(encoding="utf-8") == latest_before
    assert sorted(p.name for p in env["out_dir"].glob("step_*.pt")) == files_before


def test_force_fresh_over_existing_run_restarts_from_zero(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _config_for(env, "train.yaml", total_steps=4, decay_start_step=3)
    assert main(["train", "--config", str(cfg), "--device", "cpu"]) == 0
    capsys.readouterr()

    # --force-fresh is the explicit escape hatch: start over even though out_dir
    # already holds a run.
    assert main(["train", "--config", str(cfg), "--device", "cpu", "--force-fresh"]) == 0
    stats = _stats(capsys)
    assert stats["resumed"] is False
    assert stats["start_step"] == 0  # counts from zero, not resumed
    assert stats["step"] == 4
