#!/usr/bin/env python
"""CPU smoke-train: exercise the full train -> checkpoint -> resume path fast.

Writes tiny synthetic-but-learnable token shards, runs the *real* ``sqlpup
train`` CLI as a subprocess (a fresh run, then ``--resume auto`` with a higher
``total_steps`` via the extendable-WSD path), and asserts the model learned and
that resume genuinely continued from the checkpoint. Used by CI and runnable
locally via ``make smoke-train``; finishes in a few seconds on CPU.

Stdlib + sqlpup only. Torch is pulled solely by the CLI subprocess, never
imported here, so this script (and its unit-tested core) stay torch-free.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import yaml

from sqlpup.shard import ShardIndex, ShardWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = REPO_ROOT / "artifacts" / "smoke"
TRAIN_DIR = SMOKE_DIR / "train"
EVAL_DIR = SMOKE_DIR / "eval"
CKPT_DIR = SMOKE_DIR / "ckpt"
CONFIG = REPO_ROOT / "configs" / "train" / "smoke_cpu.yaml"
RESUME_CONFIG = SMOKE_DIR / "smoke_cpu_resume.yaml"

VOCAB_SIZE = 256
EOS_ID = VOCAB_SIZE - 1  # top id reserved for the shard separator
PATTERN_PERIOD = 16
# The resume leg extends the schedule (raising total_steps/decay_start is the
# extendable-WSD mechanism) so it trains a few genuine extra steps.
RESUME_TOTAL_STEPS = 40
RESUME_DECAY_START = 32


def build_pattern(vocab_size: int, period: int, seed: int) -> list[int]:
    """A fixed, learnable token cycle of length ``period`` drawn from the vocab.

    Deterministic in ``seed`` (same seed -> same cycle). Tokens are distinct ids
    in ``[0, vocab_size - 1)``; the top id is reserved for EOS.
    """
    if period >= vocab_size:
        raise ValueError(f"period {period} must be < vocab_size {vocab_size}")
    rng = random.Random(seed)
    return rng.sample(range(vocab_size - 1), period)


def write_synthetic_shards(
    out_dir: Path,
    *,
    documents: int,
    repeats: int,
    vocab_size: int = VOCAB_SIZE,
    eos_id: int = EOS_ID,
    period: int = PATTERN_PERIOD,
    seed: int = 0,
) -> ShardIndex:
    """Write ``documents`` docs, each the same length-``period`` cycle repeated.

    A repeating cycle is trivially next-token predictable, so even the tiny smoke
    model drives the loss well below the ``ln(vocab_size)`` random-init baseline
    within a few dozen steps. Returns the written :class:`ShardIndex`.
    """
    document = build_pattern(vocab_size, period, seed) * repeats
    writer = ShardWriter(out_dir, eos_id=eos_id, shard_size_tokens=10_000_000)
    for _ in range(documents):
        writer.add(document)
    return writer.close()


def _write_resume_config() -> None:
    """Derive the resume config from ``smoke_cpu.yaml`` with an extended schedule."""
    raw: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["total_steps"] = RESUME_TOTAL_STEPS
    raw["decay_start_step"] = RESUME_DECAY_START
    RESUME_CONFIG.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _run_train(config: Path, *, resume: str | None) -> dict[str, Any]:
    """Invoke the real CLI on ``config`` (forced to CPU) and return its stats line."""
    cmd = [
        sys.executable,
        "-m",
        "sqlpup.cli",
        "train",
        "--config",
        str(config.relative_to(REPO_ROOT)),
        "--device",
        "cpu",
    ]
    if resume is not None:
        cmd += ["--resume", resume]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    label = " ".join(cmd[3:])
    if proc.returncode != 0:
        _fail(
            f"`sqlpup {label}` exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return _parse_stats(proc.stdout, label)


def _parse_stats(stdout: str, label: str) -> dict[str, Any]:
    """Parse the last non-empty stdout line as the emitted JSON stats object."""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        _fail(f"`sqlpup {label}` printed no stats line on stdout")
    try:
        parsed: dict[str, Any] = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        _fail(f"`sqlpup {label}` last stdout line was not JSON: {lines[-1]!r} ({exc})")
    return parsed


def _check(ok: bool, message: str) -> None:
    if not ok:
        _fail(message)


def _fail(message: str) -> NoReturn:
    print(f"smoke-train FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)  # always start from a clean slate
    # Train and eval share the learnable pattern: the point is to exercise the
    # eval path and see a sensible (decreasing) loss, not to measure generalization.
    write_synthetic_shards(TRAIN_DIR, documents=32, repeats=8, seed=0)
    write_synthetic_shards(EVAL_DIR, documents=8, repeats=8, seed=0)
    _write_resume_config()

    baseline = math.log(VOCAB_SIZE)  # cross-entropy of an untrained uniform model

    fresh = _run_train(CONFIG, resume=None)
    _check(fresh["resumed"] is False, f"fresh run should not be resumed: {fresh}")
    _check(fresh["start_step"] == 0, f"fresh run should start at step 0: {fresh}")
    _check(
        fresh["final_loss"] < baseline,
        f"fresh final_loss {fresh['final_loss']:.4f} not below baseline {baseline:.4f} "
        "(model did not learn)",
    )
    _check(
        fresh["eval_loss"] < baseline,
        f"fresh eval_loss {fresh['eval_loss']:.4f} not below baseline {baseline:.4f}",
    )
    _check((CKPT_DIR / "latest.json").exists(), "latest.json missing after the fresh run")
    _check(bool(list(CKPT_DIR.glob("step_*.pt"))), "no step_*.pt checkpoints after the fresh run")
    fresh_step = int(fresh["step"])

    resumed = _run_train(RESUME_CONFIG, resume="auto")
    _check(resumed["resumed"] is True, f"resume run should be resumed: {resumed}")
    _check(
        resumed["start_step"] == fresh_step,
        f"resume should continue from step {fresh_step}, got start_step {resumed['start_step']}",
    )
    _check(
        int(resumed["step"]) > fresh_step,
        f"resume step {resumed['step']} should exceed the checkpoint step {fresh_step}",
    )
    _check(
        resumed["final_loss"] < fresh["final_loss"],
        f"resumed final_loss {resumed['final_loss']:.4f} did not improve on fresh "
        f"{fresh['final_loss']:.4f} -- the resume leg trains extra steps on a learnable "
        "pattern, so a restored run must keep decreasing",
    )

    print("smoke-train OK")
    print(f"  fresh : {json.dumps(fresh, sort_keys=True)}")
    print(f"  resume: {json.dumps(resumed, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
