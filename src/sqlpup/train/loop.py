"""The pretraining loop: AdamW + WSD, grad accumulation, eval, checkpointing.

``Trainer`` ties the pieces together and is DDP-*ready*: under ``torchrun``
(``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` set) it initializes a process group,
wraps the model in ``DistributedDataParallel``, shards data by rank, and confines
logging/eval/checkpointing to rank 0; with no such env it runs as a plain single
process with no ``torch.distributed`` side effects. Multi-GPU is only exercised
for real at AWS time, so the DDP branches are kept deliberately thin.

Device/precision policy: CUDA runs bf16 autocast (and fused AdamW); CPU and MPS
run fp32 with no autocast (bf16-on-MPS is revisited if the smoke shows it pays).
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from sqlpup.model.config import load_model_config
from sqlpup.model.transformer import SqlpupLM
from sqlpup.train.checkpoint import (
    gather_rng_states,
    load_checkpoint,
    mark_stage_final,
    prune_checkpoints,
    restore_rng_states,
    save_checkpoint,
)
from sqlpup.train.config import TrainConfig
from sqlpup.train.data import PackedDataLoader, TokenStream
from sqlpup.train.lr import wsd_lr

log = logging.getLogger(__name__)

_WANDB_EXTRA_HINT = (
    "experiment tracking is enabled (wandb_project is set) but the optional 'track' "
    "extra (wandb) is not installed.\n"
    "Install it with one of:\n"
    '    pip install "sqlpup[track]"\n'
    "    uv sync --extra track"
)


def _import_wandb() -> Any:
    """Import the optional ``wandb`` dependency, or fail loud naming the extra.

    Called only when ``wandb_project`` is set, so a run without tracking pulls
    zero wandb imports. A missing package re-raises as an :class:`ImportError`
    carrying an actionable install hint rather than a bare ``ModuleNotFoundError``.
    """
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(_WANDB_EXTRA_HINT) from exc
    return wandb


@dataclass(frozen=True, slots=True)
class DistInfo:
    """torchrun-provided distributed coordinates."""

    rank: int
    world_size: int
    local_rank: int


@dataclass(frozen=True, slots=True)
class StepLog:
    """One training-step record (kept in memory for tests / lightweight logging)."""

    step: int
    loss: float
    lr: float
    grad_norm: float
    tokens_per_s: float
    tokens: int


@dataclass(frozen=True, slots=True)
class EvalLog:
    """One evaluation record: mean loss and perplexity at a step."""

    step: int
    loss: float
    ppl: float


def dist_info_from_env(env: Mapping[str, str] | None = None) -> DistInfo | None:
    """Parse torchrun's ``RANK``/``WORLD_SIZE``/``LOCAL_RANK``; ``None`` if unset.

    All three must be present -- a partial set means the caller is not under
    ``torchrun`` and we stay single-process (no dist init).
    """
    env = os.environ if env is None else env
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= env.keys():
        return None
    return DistInfo(int(env["RANK"]), int(env["WORLD_SIZE"]), int(env["LOCAL_RANK"]))


def select_device(explicit: str | None, dist_info: DistInfo | None) -> torch.device:
    """Choose the compute device: explicit override, else cuda > mps > cpu."""
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        # Pin each rank to its local GPU under DDP.
        return torch.device(f"cuda:{dist_info.local_rank}" if dist_info else "cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_due(
    step: int,
    total_steps: int,
    last_time: float,
    now: float,
    interval_steps: int,
    interval_minutes: int,
) -> bool:
    """Whether to checkpoint after ``step``: at the end, every N steps, or by time."""
    if step >= total_steps:
        return True  # always keep a stage-final checkpoint
    if interval_steps > 0 and step % interval_steps == 0:
        return True
    return interval_minutes > 0 and (now - last_time) >= interval_minutes * 60


def compute_loss(model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """Next-token cross-entropy (mean over all tokens) for inputs ``x``, targets ``y``."""
    logits = cast(Tensor, model(x))
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def configure_optimizer(
    model: nn.Module, cfg: TrainConfig, device: torch.device
) -> torch.optim.Optimizer:
    """AdamW with the nanoGPT two-group convention.

    Tensors with ``ndim >= 2`` (matmul/embedding weights) are weight-decayed;
    ``ndim < 2`` tensors (RMSNorm gains) are not. Tied weights appear once
    because ``parameters()`` deduplicates shared tensors. ``eps`` is AdamW's
    default (1e-8); it is intentionally not a config field. Fused AdamW is the
    fast CUDA-only path.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {}
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, lr=cfg.max_lr, betas=(cfg.beta1, cfg.beta2), **kwargs)


class Trainer:
    """Owns the model, optimizer, dataloader, and the step loop."""

    def __init__(self, cfg: TrainConfig, *, device: str | None = None) -> None:
        self.cfg = cfg
        self._dist = dist_info_from_env()
        self.rank = self._dist.rank if self._dist else 0
        self.world_size = self._dist.world_size if self._dist else 1
        self._is_main = self.rank == 0
        self.device = select_device(device, self._dist)

        # Optional experiment tracking. Import at startup so a missing 'track'
        # extra fails loud before any process-group init or model allocation; an
        # unset wandb_project imports nothing. The run itself is opened lazily in
        # train() (rank 0 only) once the resumed flag is known.
        self._wandb: Any = _import_wandb() if cfg.wandb_project is not None else None
        self._wandb_run: Any = None

        self._pg_initialized = False
        if self._dist is not None:
            self._init_process_group()

        # Deterministic init: all ranks seed identically; DDP broadcasts rank 0's
        # parameters at wrap time, and the dataloader has its own (seed, epoch) RNG.
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        model_cfg = load_model_config(cfg.model_config)
        self._model_cfg = model_cfg
        if cfg.seq_len > model_cfg.max_seq_len:
            raise ValueError(
                f"seq_len {cfg.seq_len} exceeds model max_seq_len {model_cfg.max_seq_len}"
            )
        self.raw_model = SqlpupLM(model_cfg).to(self.device)
        self.model: nn.Module = self.raw_model
        # Keep a direct handle to the DDP wrapper (if any) so no_sync still works
        # when torch.compile wraps it -- self.model may then be an OptimizedModule.
        self._ddp: DistributedDataParallel | None = None
        if self._dist is not None:
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            self._ddp = DistributedDataParallel(self.raw_model, device_ids=device_ids)
            self.model = self._ddp
        if cfg.compile:
            self.model = cast(nn.Module, torch.compile(self.model))

        self.optimizer = configure_optimizer(self.raw_model, cfg, self.device)

        self.train_loader = self._build_loader(cfg.shard_index)
        self.train_iter = iter(self.train_loader)
        self.eval_loader = (
            self._build_loader(cfg.eval_shard_index) if cfg.eval_shard_index is not None else None
        )

        self.step = 0
        self.logs: list[StepLog] = []
        self.evals: list[EvalLog] = []
        self.checkpoint_steps: list[int] = []
        self._last_loss = 0.0
        self._last_ckpt_time = time.time()

    # -- construction helpers ---------------------------------------------

    def _build_loader(self, index_path: Path) -> PackedDataLoader:
        stream = TokenStream.from_index_file(index_path)
        return PackedDataLoader(
            stream,
            self.cfg.seq_len,
            self.cfg.micro_batch_size,
            self.cfg.seed,
            rank=self.rank,
            world_size=self.world_size,
        )

    def _init_process_group(self) -> None:
        backend = "nccl" if self.device.type == "cuda" else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
            self._pg_initialized = True
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

    @classmethod
    def from_checkpoint(cls, path: Path, cfg: TrainConfig, *, device: str | None = None) -> Trainer:
        """Rebuild a trainer and restore model/optimizer/step/data-position/RNG.

        ``cfg`` may raise ``total_steps``/``decay_start_step`` versus the run that
        wrote the checkpoint: that is the extendable-WSD mechanism, safe because
        :func:`wsd_lr` is a pure function of ``(step, cfg)`` with no stored state.
        """
        trainer = cls(cfg, device=device)
        ckpt = load_checkpoint(path)
        trainer.raw_model.load_state_dict(ckpt["model"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        _optimizer_state_to(trainer.optimizer, trainer.device)
        trainer.step = int(ckpt["step"])
        epoch, cursor = ckpt["data_position"]
        trainer.train_loader.seek(int(epoch), int(cursor))
        trainer.train_iter = iter(trainer.train_loader)
        restore_rng_states(ckpt["rng"])
        return trainer

    # -- precision / DDP context managers ---------------------------------

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        # CPU and MPS run fp32; bf16-on-MPS is revisited if the smoke shows gains.
        return contextlib.nullcontext()

    def _maybe_no_sync(self, skip: bool) -> contextlib.AbstractContextManager[Any]:
        # Skip gradient all-reduce on all but the last micro-step so accumulation
        # reduces once per optimizer step (correctness + a lot less comm).
        if skip and self._ddp is not None:
            return self._ddp.no_sync()
        return contextlib.nullcontext()

    # -- the step ----------------------------------------------------------

    def _train_step(self) -> tuple[float, float, float]:
        self.model.train()
        lr = wsd_lr(self.step, self.cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        accum = self.cfg.grad_accum_steps
        loss_sum = 0.0
        for micro in range(accum):
            x, y = next(self.train_iter)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with self._maybe_no_sync(micro < accum - 1):
                with self._autocast():
                    loss = compute_loss(self.model, x, y)
                (loss / accum).backward()  # type: ignore[no-untyped-call]
            loss_sum += loss.item()
        grad_norm = clip_grad_norm_(self.raw_model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return loss_sum / accum, lr, float(grad_norm)

    def train(self, *, resumed: bool = False) -> dict[str, Any]:
        """Run from the current step to ``total_steps``; return final stats.

        ``resumed`` is recorded on the tracking run (when enabled) so a resumed
        leg is distinguishable in the dashboard; it does not affect training. The
        wandb run is finished only on a clean exit (after the loop completes).
        """
        self._last_ckpt_time = time.time()
        self._start_tracking(resumed=resumed)
        while self.step < self.cfg.total_steps:
            t0 = time.perf_counter()
            loss, lr, grad_norm = self._train_step()
            dt = time.perf_counter() - t0
            self._last_loss = loss
            self._maybe_log(loss, lr, grad_norm, dt)
            self._maybe_eval()
            self._maybe_checkpoint()
        self._finish_tracking()
        return self._final_stats()

    # -- periodic actions (rank 0 only for log/eval-record/checkpoint) -----

    def _maybe_log(self, loss: float, lr: float, grad_norm: float, dt: float) -> None:
        interval = self.cfg.log_interval_steps
        if not (self._is_main and interval > 0 and self.step % interval == 0):
            return
        tokens = self.cfg.effective_tokens_per_step(self.world_size)
        tokens_per_s = tokens / dt if dt > 0 else 0.0
        self.logs.append(StepLog(self.step, loss, lr, grad_norm, tokens_per_s, tokens))
        log.info(
            "step %d | loss %.4f | lr %.3e | grad_norm %.3f | %.0f tok/s",
            self.step,
            loss,
            lr,
            grad_norm,
            tokens_per_s,
        )
        if self._wandb_run is not None:
            self._wandb.log(
                {
                    "train/loss": loss,
                    "lr": lr,
                    "tokens_seen": tokens * self.step,
                    "tokens_per_s": tokens_per_s,
                },
                step=self.step,
            )

    def _maybe_eval(self) -> None:
        interval = self.cfg.eval_interval_steps
        if self.eval_loader is None or interval <= 0 or self.step % interval != 0:
            return
        loss, ppl = self.evaluate()
        if self._is_main:
            self.evals.append(EvalLog(self.step, loss, ppl))
            log.info("eval step %d | loss %.4f | ppl %.3f", self.step, loss, ppl)
            if self._wandb_run is not None:
                self._wandb.log({"eval/loss": loss, "eval/ppl": ppl}, step=self.step)

    def _maybe_checkpoint(self) -> None:
        if not self._is_main:
            return
        now = time.time()
        if not checkpoint_due(
            self.step,
            self.cfg.total_steps,
            self._last_ckpt_time,
            now,
            self.cfg.checkpoint_interval_steps,
            self.cfg.checkpoint_interval_minutes,
        ):
            return
        is_final = self.step >= self.cfg.total_steps
        save_checkpoint(
            self.cfg.out_dir,
            self.step,
            self.raw_model,
            self.optimizer,
            self._config_snapshot(),
            self.train_loader.position(),
            gather_rng_states(),
        )
        if is_final:
            mark_stage_final(self.cfg.out_dir, self.step)
        prune_checkpoints(self.cfg.out_dir, self.cfg.keep_checkpoints)
        self.checkpoint_steps.append(self.step)
        self._last_ckpt_time = now

    def evaluate(self) -> tuple[float, float]:
        """Mean eval loss and perplexity over ``eval_batches`` (all-reduced under DDP).

        Deterministic and comparable across calls: the eval loader is reset to
        the start of epoch 0 each time.
        """
        assert self.eval_loader is not None
        self.model.eval()
        self.eval_loader.seek(0, 0)
        eval_iter = iter(self.eval_loader)
        losses: list[Tensor] = []
        with torch.no_grad():
            for _ in range(self.cfg.eval_batches):
                x, y = next(eval_iter)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with self._autocast():
                    losses.append(compute_loss(self.model, x, y))
        mean = torch.stack(losses).mean()
        if self._dist is not None:
            # AVG is not supported on gloo; SUM/world_size works everywhere.
            dist.all_reduce(mean, op=dist.ReduceOp.SUM)
            mean = mean / self.world_size
        self.model.train()
        return float(mean.item()), float(torch.exp(mean).item())

    # -- experiment tracking (optional wandb; rank 0 only) -----------------

    def _start_tracking(self, *, resumed: bool) -> None:
        """Open the wandb run when tracking is enabled. Rank 0 only."""
        if self._wandb is None or not self._is_main:
            return
        name = self.cfg.wandb_run_name or self._default_run_name()
        # No entity kwarg: it comes from the API key's default team/user.
        self._wandb_run = self._wandb.init(
            project=self.cfg.wandb_project,
            name=name,
            config=self._tracking_config(resumed=resumed),
        )

    def _default_run_name(self) -> str:
        """Fallback run name: the out_dir basename plus a UTC timestamp."""
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"{self.cfg.out_dir.name}-{stamp}"

    def _tracking_config(self, *, resumed: bool) -> dict[str, Any]:
        """Flattened train+model config recorded on the run (plus the resumed flag)."""
        config = dict(self._config_snapshot())
        config.update({f"model_{k}": v for k, v in asdict(self._model_cfg).items()})
        config["resumed"] = resumed
        config["world_size"] = self.world_size
        return config

    def _finish_tracking(self) -> None:
        """Close the wandb run on a clean exit. Rank 0 only (no-op otherwise)."""
        if self._wandb_run is not None:
            self._wandb.finish()
            self._wandb_run = None

    # -- misc --------------------------------------------------------------

    def _config_snapshot(self) -> dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self.cfg).items()}

    def _final_stats(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "final_loss": self._last_loss,
            "final_lr": wsd_lr(max(self.step - 1, 0), self.cfg),
            "effective_tokens_per_step": self.cfg.effective_tokens_per_step(self.world_size),
            "num_logs": len(self.logs),
            "num_evals": len(self.evals),
            "num_checkpoints": len(self.checkpoint_steps),
            "world_size": self.world_size,
        }

    def close(self) -> None:
        """Tear down the process group if this trainer created one."""
        if self._pg_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._pg_initialized = False

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _optimizer_state_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    # Optimizer state loads on CPU (checkpoints are map_location='cpu'); move the
    # moment buffers onto the compute device so step() matches the params.
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
