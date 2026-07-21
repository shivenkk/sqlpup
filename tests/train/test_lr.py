from pathlib import Path

import pytest

from sqlpup.train.config import TrainConfig
from sqlpup.train.lr import wsd_lr


def make_cfg(**overrides: object) -> TrainConfig:
    params: dict[str, object] = {
        "model_config": Path("m.yaml"),
        "shard_index": Path("i.json"),
        "out_dir": Path("out"),
        "micro_batch_size": 1,
        "grad_accum_steps": 1,
        "max_lr": 1.0,
        "min_lr_ratio": 0.1,
        "warmup_steps": 10,
        "total_steps": 200,
        "decay_start_step": 100,
        # These fields are irrelevant to the LR schedule; set to their minimum
        # valid values (TrainConfig now rejects non-positive intervals/counts).
        "eval_interval_steps": 1,
        "eval_batches": 1,
        "checkpoint_interval_steps": 1,
        "log_interval_steps": 1,
        "seed": 0,
    }
    params.update(overrides)
    return TrainConfig(**params)  # type: ignore[arg-type]


def test_warmup_start_is_zero() -> None:
    # 0-indexed warmup: step 0 has lr 0, ramps linearly to max_lr at warmup_steps.
    assert wsd_lr(0, make_cfg()) == pytest.approx(0.0)


def test_warmup_is_linear() -> None:
    cfg = make_cfg()
    assert wsd_lr(5, cfg) == pytest.approx(0.5)  # 5/10 * max_lr


def test_end_of_warmup_hits_max() -> None:
    assert wsd_lr(10, make_cfg()) == pytest.approx(1.0)


def test_plateau_is_constant_max() -> None:
    cfg = make_cfg()
    assert wsd_lr(11, cfg) == pytest.approx(1.0)
    assert wsd_lr(50, cfg) == pytest.approx(1.0)
    assert wsd_lr(99, cfg) == pytest.approx(1.0)


def test_decay_start_is_max() -> None:
    assert wsd_lr(100, make_cfg()) == pytest.approx(1.0)


def test_decay_midpoint() -> None:
    # halfway through the [100, 200] tail -> halfway between max and min.
    cfg = make_cfg()
    assert wsd_lr(150, cfg) == pytest.approx(0.55)  # 1 - 0.9 * 0.5


def test_total_step_hits_min() -> None:
    cfg = make_cfg()
    assert wsd_lr(200, cfg) == pytest.approx(0.1)  # min_lr = max_lr * min_lr_ratio


def test_beyond_total_clamps_at_min() -> None:
    cfg = make_cfg()
    assert wsd_lr(500, cfg) == pytest.approx(0.1)


def test_zero_warmup_no_division_error() -> None:
    cfg = make_cfg(warmup_steps=0, decay_start_step=50, total_steps=100)
    assert wsd_lr(0, cfg) == pytest.approx(1.0)  # straight onto the plateau


def test_no_decay_tail_no_division_error() -> None:
    # decay_start == total: plateau to the end, then clamp at min.
    cfg = make_cfg(warmup_steps=10, decay_start_step=100, total_steps=100)
    assert wsd_lr(99, cfg) == pytest.approx(1.0)
    assert wsd_lr(100, cfg) == pytest.approx(0.1)


def test_lr_never_below_min_or_above_max() -> None:
    cfg = make_cfg()
    for step in range(0, 260):
        lr = wsd_lr(step, cfg)
        assert 0.0 <= lr <= 1.0 + 1e-12


def test_wsd_extendable_same_lr_on_plateau() -> None:
    # The extendability contract: on a mid-plateau step, raising decay_start /
    # total_steps (a resume-time extension) leaves the current lr unchanged,
    # because the schedule is a pure function of (step, config) with no state.
    short = make_cfg(warmup_steps=10, decay_start_step=100, total_steps=200)
    extended = make_cfg(warmup_steps=10, decay_start_step=300, total_steps=600)
    for step in (10, 50, 99):  # steps on the plateau of both schedules
        assert wsd_lr(step, short) == pytest.approx(wsd_lr(step, extended))
