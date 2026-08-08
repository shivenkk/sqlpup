"""End-to-end GRPO: a real trainer, a real reward, a real optimizer step.

Every other test here mocks the boundary this one exercises. The failures that
have actually cost this project time were all integration failures that unit
tests passed straight through -- a constrained decoder that wedged, a hardcoded
pad id, a reward contract mismatch. So this drives an actual
``GRPOTrainer.train()`` on a tiny randomly-initialised model, on CPU, and only
asserts that the machine turns over: rewards were computed for real rollouts
and a step completed.

It is deliberately not a quality test. A random 2-layer model learns nothing,
and asserting otherwise would be asserting noise.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from sqlpup.model.config import ModelConfig  # noqa: E402
from sqlpup.model.export import write_hf_directory  # noqa: E402
from sqlpup.model.transformer import SqlpupLM  # noqa: E402
from sqlpup.rl.train import run_grpo  # noqa: E402


@pytest.fixture
def tiny_model_dir(tmp_path: Path) -> Path:
    tokenizer = Path("artifacts/tokenizer/tokenizer.json")
    if not tokenizer.exists():
        pytest.skip("project tokenizer not present in this checkout")
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        vocab_size=32768,
        max_seq_len=512,
    )
    model = SqlpupLM(cfg)
    out = tmp_path / "tiny"
    write_hf_directory(
        out, state_dict=model.state_dict(), cfg=cfg, dtype="fp32", tokenizer_path=tokenizer
    )
    return out


@pytest.fixture
def rows(tmp_path: Path) -> list[dict[str, str]]:
    db_dir = tmp_path / "dbs" / "toy"
    db_dir.mkdir(parents=True)
    db = db_dir / "toy.sqlite"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE t (id INTEGER); INSERT INTO t (id) VALUES (1), (2);")
    con.commit()
    con.close()
    return [
        {
            "prompt": "-- schema: t(id)\n-- question: how many?\n",
            "gold_sql": "SELECT id FROM t",
            "db_path": str(db),
        },
        {
            "prompt": "-- schema: t(id)\n-- question: list ids\n",
            "gold_sql": "SELECT id FROM t",
            "db_path": str(db),
        },
    ]


@pytest.mark.slow
def test_one_grpo_step_completes_and_scores_real_rollouts(
    tiny_model_dir: Path, rows: list[dict[str, str]], tmp_path: Path
) -> None:
    receipt = run_grpo(
        model_dir=tiny_model_dir,
        rows=rows,
        out_dir=tmp_path / "grpo-out",
        num_generations=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_completion_length=16,
        context_limit=512,
        max_steps=1,
        device="cpu",
    )
    assert receipt["steps"] >= 1
    assert receipt["rollouts_scored"] > 0
    # A random model produces garbage SQL, so near-zero reward is correct here;
    # what must hold is that the reward ran without erroring on every row.
    assert receipt["reward_errors"] == 0
    assert receipt["halt_reason"] is None
    assert (tmp_path / "grpo-out").exists()
