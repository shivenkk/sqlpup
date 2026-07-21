import json
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from sqlpup.model.config import ModelConfig
from sqlpup.model.transformer import SqlpupLM
from sqlpup.train.checkpoint import (
    gather_rng_states,
    load_checkpoint,
    mark_stage_final,
    prune_checkpoints,
    restore_rng_states,
    save_checkpoint,
)

CFG = ModelConfig(
    d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, vocab_size=64, max_seq_len=32
)


def _model_and_opt() -> tuple[SqlpupLM, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = SqlpupLM(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


def _take_a_step(model: SqlpupLM, opt: torch.optim.Optimizer) -> None:
    tokens = torch.randint(0, CFG.vocab_size, (2, 8))
    targets = torch.randint(0, CFG.vocab_size, (2, 8))
    loss = torch.nn.functional.cross_entropy(
        model(tokens).reshape(-1, CFG.vocab_size), targets.reshape(-1)
    )
    loss.backward()  # type: ignore[no-untyped-call]
    opt.step()
    opt.zero_grad()


def _save(tmp_path: Path, step: int) -> Path:
    model, opt = _model_and_opt()
    _take_a_step(model, opt)
    return save_checkpoint(
        tmp_path, step, model, opt, {"note": "snap"}, (0, 0), gather_rng_states()
    )


def test_save_load_roundtrip_bit_equal(tmp_path: Path) -> None:
    model_a, opt_a = _model_and_opt()
    _take_a_step(model_a, opt_a)
    save_checkpoint(tmp_path, 1, model_a, opt_a, {"n": 1}, (2, 6), gather_rng_states())

    ckpt = load_checkpoint(tmp_path / "step_1.pt")
    assert ckpt["step"] == 1
    assert ckpt["data_position"] == (2, 6)

    model_b, opt_b = _model_and_opt()  # same init, but we overwrite from ckpt
    model_b.load_state_dict(ckpt["model"])
    opt_b.load_state_dict(ckpt["optimizer"])

    for (name, pa), (_, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert torch.equal(pa, pb), name
    # optimizer moment buffers restored bit-exactly
    for pa in model_a.parameters():
        sa = opt_a.state[pa]
        # match by identical shape+values in restored optimizer
        found = any(
            torch.equal(sa["exp_avg"], sb.get("exp_avg", torch.empty(0)))
            for sb in opt_b.state.values()
            if "exp_avg" in sb and sb["exp_avg"].shape == sa["exp_avg"].shape
        )
        assert found


def test_atomic_leaves_no_tmp(tmp_path: Path) -> None:
    _save(tmp_path, 3)
    assert (tmp_path / "step_3.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["step"] == 3
    assert latest["path"] == "step_3.pt"
    assert "wall_time" in latest


def test_sweep_removes_stale_tmp_only(tmp_path: Path) -> None:
    from sqlpup.train.checkpoint import _sweep_stale_tmp

    _save(tmp_path, 10)  # a real step_10.pt checkpoint + latest.json
    real = tmp_path / "step_10.pt"
    latest = tmp_path / "latest.json"
    real_bytes = real.read_bytes()
    latest_bytes = latest.read_bytes()

    # Two orphans left by crashed/spot-killed saves, plus the tmp the current
    # save is about to write (not stale -- must be preserved).
    (tmp_path / "step_7.pt.tmp").write_bytes(b"orphan-7")
    (tmp_path / "step_9.pt.tmp").write_bytes(b"orphan-9")
    keep = tmp_path / "step_20.pt.tmp"
    keep.write_bytes(b"in-flight")

    _sweep_stale_tmp(tmp_path, keep)

    assert not (tmp_path / "step_7.pt.tmp").exists()
    assert not (tmp_path / "step_9.pt.tmp").exists()
    assert keep.read_bytes() == b"in-flight"  # never sweeps the file about to be written
    assert real.read_bytes() == real_bytes  # real checkpoint byte-untouched
    assert latest.read_bytes() == latest_bytes  # latest.json byte-untouched


def test_save_sweeps_orphan_tmp_and_stays_atomic(tmp_path: Path) -> None:
    _save(tmp_path, 10)
    real_bytes = (tmp_path / "step_10.pt").read_bytes()
    (tmp_path / "step_5.pt.tmp").write_bytes(b"orphan")
    (tmp_path / "step_9.pt.tmp").write_bytes(b"orphan")

    _save(tmp_path, 20)  # a fresh successful save sweeps orphans before writing

    assert (tmp_path / "step_20.pt").exists()  # new checkpoint present
    assert not list(tmp_path.glob("*.tmp"))  # orphans swept + save still atomic
    assert (tmp_path / "step_10.pt").read_bytes() == real_bytes  # real ckpt untouched
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["step"] == 20


def test_latest_tracks_newest(tmp_path: Path) -> None:
    _save(tmp_path, 5)
    _save(tmp_path, 10)
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["step"] == 10
    assert latest["path"] == "step_10.pt"


def test_load_from_dir_uses_latest(tmp_path: Path) -> None:
    _save(tmp_path, 5)
    _save(tmp_path, 10)
    ckpt = load_checkpoint(tmp_path)  # a directory -> resolves via latest.json
    assert ckpt["step"] == 10


def test_prune_keeps_last_n(tmp_path: Path) -> None:
    for step in (1, 2, 3, 4, 5):
        _save(tmp_path, step)
    prune_checkpoints(tmp_path, keep_last=3)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_3.pt", "step_4.pt", "step_5.pt"]


def test_prune_sorts_numerically_not_lexically(tmp_path: Path) -> None:
    for step in (2, 9, 10, 11):
        _save(tmp_path, step)
    prune_checkpoints(tmp_path, keep_last=2)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_10.pt", "step_11.pt"]  # not step_2/step_9 by string


def test_prune_protects_stage_final(tmp_path: Path) -> None:
    for step in (1, 2, 3, 4, 5):
        _save(tmp_path, step)
    mark_stage_final(tmp_path, 1)  # protect the oldest
    prune_checkpoints(tmp_path, keep_last=2)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_1.pt", "step_4.pt", "step_5.pt"]


def test_rng_states_roundtrip(tmp_path: Path) -> None:
    states = gather_rng_states()
    t_before = torch.rand(4)
    n_before = np.random.rand(4)
    p_before = [random.random() for _ in range(4)]

    restore_rng_states(states)
    assert torch.equal(torch.rand(4), t_before)
    assert np.array_equal(np.random.rand(4), n_before)
    assert [random.random() for _ in range(4)] == p_before


def test_restore_rng_states_tolerates_world_size_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A checkpoint saved on a 4-GPU box must restore on a 1-GPU box: only the
    # per-device states that fit are applied. set_rng_state_all raises when the
    # checkpoint carries more states than the machine has devices, so it must
    # never be used here.
    applied: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device=0: applied.append(int(device)),
    )

    def _explode(states: object) -> None:
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", _explode)

    states = gather_rng_states()
    states["cuda"] = [torch.get_rng_state()] * 4  # snapshot from a 4-GPU world
    restore_rng_states(states)
    assert applied == [0]
