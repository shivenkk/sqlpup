from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from sqlpup.model.transformer import SqlpupLM
from sqlpup.shard import ShardWriter
from sqlpup.train.config import TrainConfig
from sqlpup.train.loop import (
    Trainer,
    checkpoint_due,
    compute_loss,
    configure_optimizer,
    dist_info_from_env,
    select_device,
)

PATTERN = list(range(1, 13))  # a short, easily-learnable repeating cycle
EOS = 0


def write_model_config(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(
        "d_model: 32\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nd_ff: 64\n"
        "vocab_size: 64\nmax_seq_len: 64\n"
    )
    return path


def build_shards(shard_dir: Path, repeats: int) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(shard_dir, eos_id=EOS, shard_size_tokens=10_000)
    writer.add(PATTERN * repeats)
    writer.close()
    return shard_dir / "index.json"


def make_train_config(
    model_cfg: Path, shard_index: Path, out_dir: Path, **overrides: Any
) -> TrainConfig:
    params: dict[str, Any] = {
        "model_config": model_cfg,
        "shard_index": shard_index,
        "out_dir": out_dir,
        "micro_batch_size": 2,
        "grad_accum_steps": 1,
        "max_lr": 1e-2,
        "warmup_steps": 2,
        "total_steps": 10,
        "decay_start_step": 10,
        # Eval is gated off by eval_shard_index=None below; these just need to
        # be valid (TrainConfig rejects non-positive intervals/counts).
        "eval_interval_steps": 1,
        "eval_batches": 1,
        "checkpoint_interval_steps": 1000,
        "log_interval_steps": 1,
        "seed": 1234,
        "seq_len": 8,
    }
    params.update(overrides)
    return TrainConfig(**params)


# --- pure DDP / scheduling helpers ---------------------------------------


def test_dist_info_from_env_absent() -> None:
    assert dist_info_from_env({}) is None
    assert dist_info_from_env({"RANK": "0"}) is None  # needs all three


def test_dist_info_from_env_present() -> None:
    info = dist_info_from_env({"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "1"})
    assert info is not None
    assert (info.rank, info.world_size, info.local_rank) == (3, 8, 1)


def test_select_device_explicit() -> None:
    assert select_device("cpu", None).type == "cpu"


def test_select_device_auto_is_valid() -> None:
    assert select_device(None, None).type in {"cuda", "mps", "cpu"}


def test_checkpoint_due_logic() -> None:
    # always at total_steps
    assert checkpoint_due(10, 10, last_time=0.0, now=0.0, interval_steps=1000, interval_minutes=30)
    # step interval
    assert checkpoint_due(5, 10, last_time=0.0, now=0.0, interval_steps=5, interval_minutes=30)
    assert not checkpoint_due(4, 10, last_time=0.0, now=0.0, interval_steps=5, interval_minutes=30)
    # time interval (31 minutes elapsed vs a 30-minute interval)
    hour = 31 * 60
    assert checkpoint_due(4, 10, last_time=0.0, now=hour, interval_steps=1000, interval_minutes=30)
    assert not checkpoint_due(
        4, 10, last_time=0.0, now=60.0, interval_steps=1000, interval_minutes=30
    )


# --- optimizer configuration ---------------------------------------------


def test_optimizer_two_param_groups_by_ndim(tmp_path: Path) -> None:
    from sqlpup.model.config import load_model_config

    cfg = make_train_config(write_model_config(tmp_path), tmp_path / "x", tmp_path / "o")
    model = SqlpupLM(load_model_config(cfg.model_config))
    opt = configure_optimizer(model, cfg, torch.device("cpu"))
    assert len(opt.param_groups) == 2
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == pytest.approx(cfg.weight_decay)
    assert no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])  # matmuls/embeddings decay
    assert all(p.ndim < 2 for p in no_decay["params"])  # norms do not
    # RMSNorm weights: one per block input + post-attn, plus the final norm.
    assert len(no_decay["params"]) == 2 * 2 + 1


# --- gradient-accumulation equivalence -----------------------------------


def test_grad_accum_equivalence() -> None:
    from sqlpup.model.config import ModelConfig

    cfg = ModelConfig(
        d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, vocab_size=64, max_seq_len=32
    )
    x = torch.randint(0, 64, (4, 8))
    y = torch.randint(0, 64, (4, 8))

    torch.manual_seed(0)
    model_full = SqlpupLM(cfg)
    torch.manual_seed(0)
    model_accum = SqlpupLM(cfg)

    # accum = 1 x (batch 4): one mean-loss backward.
    compute_loss(model_full, x, y).backward()  # type: ignore[no-untyped-call]

    # accum = 2 x (batch 2): each micro-loss divided by grad_accum, summed.
    (compute_loss(model_accum, x[:2], y[:2]) / 2).backward()  # type: ignore[no-untyped-call]
    (compute_loss(model_accum, x[2:], y[2:]) / 2).backward()  # type: ignore[no-untyped-call]

    for (name, pf), (_, pa) in zip(
        model_full.named_parameters(), model_accum.named_parameters(), strict=True
    ):
        assert pf.grad is not None and pa.grad is not None
        # fp32 CPU: the only difference is cross_entropy reduction order, so the
        # tolerance is tight.
        torch.testing.assert_close(pf.grad, pa.grad, rtol=1e-5, atol=1e-6, msg=name)


# --- learning actually happens -------------------------------------------


def test_loss_decreases(tmp_path: Path) -> None:
    model_cfg = write_model_config(tmp_path)
    index = build_shards(tmp_path / "shards", repeats=200)
    cfg = make_train_config(
        model_cfg,
        index,
        tmp_path / "run",
        total_steps=40,
        warmup_steps=5,
        decay_start_step=40,
        micro_batch_size=4,
        seq_len=16,
        log_interval_steps=1,
        max_lr=1e-2,
    )
    trainer = Trainer(cfg, device="cpu")
    trainer.train()
    first, last = trainer.logs[0].loss, trainer.logs[-1].loss
    assert last < 0.8 * first  # a clear (>20%) drop on a learnable pattern


# --- resume is bit-exact --------------------------------------------------


def _final_params(trainer: Trainer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in trainer.raw_model.parameters()]


def test_resume_exactness(tmp_path: Path) -> None:
    model_cfg = write_model_config(tmp_path)
    index = build_shards(tmp_path / "shards", repeats=10)  # small: resume crosses epochs
    n = 6
    common = {
        "micro_batch_size": 2,
        "grad_accum_steps": 2,
        "seq_len": 8,
        "warmup_steps": 2,
        "max_lr": 5e-3,
        "log_interval_steps": 1000,
        "checkpoint_interval_steps": 1000,  # only the final checkpoint fires
    }
    # Plateau schedule so lr in [0, n) matches between the short and extended
    # configs -- this isolates state restoration from schedule differences.
    cfg_2n = make_train_config(
        model_cfg, index, tmp_path / "cont", total_steps=2 * n, decay_start_step=2 * n, **common
    )
    cfg_n = make_train_config(
        model_cfg, index, tmp_path / "split", total_steps=n, decay_start_step=n, **common
    )

    continuous = Trainer(cfg_2n, device="cpu")
    continuous.train()
    want = _final_params(continuous)

    first_half = Trainer(cfg_n, device="cpu")
    first_half.train()  # saves stage-final at step n

    # The passed cfg raises total_steps/decay_start -- the extendable-WSD path.
    resumed = Trainer.from_checkpoint(tmp_path / "split", cfg_2n, device="cpu")
    assert resumed.step == n
    resumed.train()
    got = _final_params(resumed)

    for pc, pr in zip(want, got, strict=True):
        torch.testing.assert_close(pc, pr, rtol=0, atol=0)  # bit-identical


# --- smoke: intervals fire, accounting is right ---------------------------


def test_smoke_intervals_and_token_accounting(tmp_path: Path) -> None:
    model_cfg = write_model_config(tmp_path)
    index = build_shards(tmp_path / "shards", repeats=50)
    eval_index = build_shards(tmp_path / "eval", repeats=20)
    out_dir = tmp_path / "run"
    cfg = make_train_config(
        model_cfg,
        index,
        out_dir,
        eval_shard_index=eval_index,
        total_steps=10,
        decay_start_step=10,
        micro_batch_size=2,
        grad_accum_steps=1,
        seq_len=8,
        log_interval_steps=2,
        eval_interval_steps=5,
        eval_batches=3,
        checkpoint_interval_steps=5,
    )
    trainer = Trainer(cfg, device="cpu")
    stats = trainer.train()

    assert stats["effective_tokens_per_step"] == 2 * 1 * 8 * 1
    assert stats["num_logs"] == 5  # steps 2, 4, 6, 8, 10
    assert stats["num_evals"] == 2  # steps 5, 10
    assert trainer.checkpoint_steps == [5, 10]  # steps + final
    assert (out_dir / "step_10.pt").exists()
    latest = json.loads((out_dir / "latest.json").read_text())
    assert latest["stage_final"] == "step_10.pt"  # final marked
    # eval produced a finite perplexity
    assert trainer.evals[-1].ppl > 0


def test_single_process_does_not_init_dist(tmp_path: Path) -> None:
    model_cfg = write_model_config(tmp_path)
    index = build_shards(tmp_path / "shards", repeats=20)
    cfg = make_train_config(model_cfg, index, tmp_path / "run")
    trainer = Trainer(cfg, device="cpu")
    assert trainer.world_size == 1
    assert trainer.rank == 0
    assert not torch.distributed.is_initialized()  # no side effects without env
    trainer.close()  # no-op, must not raise


# --- optional wandb tracking ---------------------------------------------


class _FakeWandb:
    """Stand-in for the ``wandb`` module that records init/log/finish calls.

    Injected into ``sys.modules["wandb"]`` so the loop's lazy ``import wandb``
    resolves to it -- the tests stay fully offline and never touch real wandb.
    """

    def __init__(self) -> None:
        self.init_kwargs: list[dict[str, Any]] = []
        self.logs: list[tuple[dict[str, Any], int | None]] = []
        self.finish_count = 0

    def init(self, **kwargs: Any) -> _FakeWandb:
        self.init_kwargs.append(kwargs)
        return self  # doubles as the returned run handle

    def log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        self.logs.append((data, step))

    def finish(self) -> None:
        self.finish_count += 1


def _install_fake_wandb(monkeypatch: pytest.MonkeyPatch) -> _FakeWandb:
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_wandb_disabled_imports_and_calls_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fake is present in sys.modules, but with no wandb_project the loop must
    # neither import nor call it (current behavior, byte-for-byte).
    fake = _install_fake_wandb(monkeypatch)
    model_cfg = write_model_config(tmp_path)
    index = build_shards(tmp_path / "shards", repeats=50)
    cfg = make_train_config(model_cfg, index, tmp_path / "run", total_steps=4, decay_start_step=4)
    trainer = Trainer(cfg, device="cpu")
    assert trainer._wandb is None
    trainer.train()
    assert fake.init_kwargs == []
    assert fake.logs == []
    assert fake.finish_count == 0


def test_wandb_enabled_init_log_cadence_and_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install_fake_wandb(monkeypatch)
    model_cfg = write_model_config(tmp_path)
    index = build_shards(tmp_path / "shards", repeats=50)
    eval_index = build_shards(tmp_path / "eval", repeats=20)
    out_dir = tmp_path / "run"
    cfg = make_train_config(
        model_cfg,
        index,
        out_dir,
        eval_shard_index=eval_index,
        total_steps=10,
        decay_start_step=10,
        micro_batch_size=2,
        grad_accum_steps=1,
        seq_len=8,
        log_interval_steps=2,
        eval_interval_steps=5,
        eval_batches=2,
        checkpoint_interval_steps=1000,
        wandb_project="sqlpup-proxy",
        wandb_run_name="unit-run",
    )
    Trainer(cfg, device="cpu").train(resumed=False)

    # init once, with project/name and a flattened train+model config.
    assert len(fake.init_kwargs) == 1
    init = fake.init_kwargs[0]
    assert init["project"] == "sqlpup-proxy"
    assert init["name"] == "unit-run"
    assert "entity" not in init  # entity comes from the API key default
    config = init["config"]
    assert config["resumed"] is False
    assert config["max_lr"] == cfg.max_lr
    assert config["seq_len"] == 8
    assert config["model_d_model"] == 32  # model config flattened under model_*
    assert isinstance(config["shard_index"], str)  # Paths serialized as strings

    # cadence: train metrics every log interval, eval metrics every eval interval.
    train_logs = [(d, s) for d, s in fake.logs if "train/loss" in d]
    eval_logs = [(d, s) for d, s in fake.logs if "eval/loss" in d]
    assert [s for _, s in train_logs] == [2, 4, 6, 8, 10]
    assert [s for _, s in eval_logs] == [5, 10]
    first_train = train_logs[0][0]
    assert set(first_train) == {"train/loss", "lr", "tokens_seen", "tokens_per_s"}
    assert first_train["tokens_seen"] == cfg.effective_tokens_per_step() * 2  # cumulative
    assert set(eval_logs[0][0]) == {"eval/loss", "eval/ppl"}
    assert fake.finish_count == 1


def test_wandb_run_name_defaults_to_out_dir_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install_fake_wandb(monkeypatch)
    model_cfg = write_model_config(tmp_path)
    index = build_shards(tmp_path / "shards", repeats=50)
    out_dir = tmp_path / "proxy_30m"
    cfg = make_train_config(
        model_cfg,
        index,
        out_dir,
        total_steps=2,
        decay_start_step=2,
        log_interval_steps=1000,
        checkpoint_interval_steps=1000,
        wandb_project="p",
    )
    Trainer(cfg, device="cpu").train()
    assert fake.init_kwargs[0]["name"].startswith("proxy_30m")  # basename + timestamp


def test_wandb_resumed_flag_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_wandb(monkeypatch)
    model_cfg = write_model_config(tmp_path)
    index = build_shards(tmp_path / "shards", repeats=50)
    cfg = make_train_config(
        model_cfg,
        index,
        tmp_path / "run",
        total_steps=2,
        decay_start_step=2,
        log_interval_steps=1000,
        checkpoint_interval_steps=1000,
        wandb_project="p",
    )
    Trainer(cfg, device="cpu").train(resumed=True)
    assert fake.init_kwargs[0]["config"]["resumed"] is True


def test_wandb_missing_fails_loud_naming_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A None entry makes `import wandb` raise ImportError, simulating the extra
    # being absent regardless of the environment.
    monkeypatch.setitem(sys.modules, "wandb", None)
    model_cfg = write_model_config(tmp_path)
    index = build_shards(tmp_path / "shards", repeats=10)
    cfg = make_train_config(model_cfg, index, tmp_path / "run", wandb_project="p")
    with pytest.raises(ImportError, match="track"):
        Trainer(cfg, device="cpu")
