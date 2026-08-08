"""Config validation and the trainer-side guard.

These checks exist to fail on a laptop in milliseconds rather than on a rented
GPU ten minutes into a boot. Every one of them corresponds to a constraint TRL
enforces at runtime or a mistake that would silently waste the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

trl = pytest.importorskip("trl")

from sqlpup.rl.setup import RolloutGuard, build_grpo_config  # noqa: E402


def test_effective_batch_must_divide_by_num_generations() -> None:
    """TRL's documented constraint: effective batch (processes x per-device x
    accumulation) must be a multiple of num_generations, else it raises deep
    inside the first training step."""
    with pytest.raises(ValueError, match="divisible"):
        build_grpo_config(
            out_dir="/tmp/x",
            num_generations=4,
            per_device_train_batch_size=3,
            gradient_accumulation_steps=1,
        )


def test_a_valid_geometry_is_accepted() -> None:
    cfg = build_grpo_config(
        out_dir="/tmp/x",
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
    )
    assert cfg.num_generations == 4


def test_extra_dataset_columns_are_never_stripped() -> None:
    """The reward reads gold_sql and db_path off the row. If TRL drops them the
    reward returns zero for everything and the run trains on noise."""
    cfg = build_grpo_config(out_dir="/tmp/x")
    assert cfg.remove_unused_columns is False


def test_completion_budget_must_fit_the_context() -> None:
    with pytest.raises(ValueError, match="context"):
        build_grpo_config(out_dir="/tmp/x", max_completion_length=4096, context_limit=2048)


def test_vllm_stays_off_by_default() -> None:
    """vLLM 0.26 hard-pins torch==2.11 against our 2.13, and rollouts for a
    394M model are cheap enough without it. Keeping it off removes the single
    largest integration risk from the pilot."""
    cfg = build_grpo_config(out_dir="/tmp/x")
    assert cfg.use_vllm is False


def test_the_guard_stops_training_when_health_trips() -> None:
    from transformers import TrainerControl, TrainerState

    guard = RolloutGuard()
    control = TrainerControl()
    state = TrainerState()

    for _ in range(6):  # establish a length baseline
        guard.on_log(None, state, control, logs={"completions/mean_length": 100.0})
    still_running = not control.should_training_stop
    assert still_running

    for _ in range(4):
        guard.on_log(None, state, control, logs={"completions/mean_length": 2.0})
    assert control.should_training_stop
    reason = guard.halt_reason
    assert reason is not None and "length" in reason.lower()


def test_the_guard_is_inert_on_a_healthy_run() -> None:
    from transformers import TrainerControl, TrainerState

    guard = RolloutGuard()
    control = TrainerControl()
    for _ in range(30):
        guard.on_log(
            None,
            TrainerState(),
            control,
            logs={
                "completions/mean_length": 90.0,
                "frac_reward_zero_std": 0.3,
                "completions/clipped_ratio": 0.02,
            },
        )
    assert control.should_training_stop is False
    assert guard.halt_reason is None


def test_the_receipt_is_written_to_its_own_file_not_stdout(tmp_path: Path) -> None:
    """Measured on the first pilot: the box captured stdout into receipt.json
    and got a 20MB file, because TRL logs per-step metrics and a rich
    completions table to stdout too. The run's own summary was buried at the
    end and the artefact would not parse as JSON. A receipt nobody can read is
    not a receipt."""
    from sqlpup.rl.train import write_receipt

    out = tmp_path / "nested" / "receipt.json"
    write_receipt(out, {"steps": 60, "halt_reason": None})
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["steps"] == 60
    assert parsed["halt_reason"] is None
