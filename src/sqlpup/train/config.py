"""Typed configuration for the pretraining loop.

Mirrors the loader conventions in :mod:`sqlpup.config` and
:mod:`sqlpup.model.config` (frozen slotted dataclass, strict YAML parsing with
unknown/missing-key rejection, shared ``_load_yaml``/``_check_keys`` helpers).
Kept free of ``torch`` so it, and the learning-rate schedule that depends on
it, import at ``sqlpup`` package-import time without pulling the optional
``train`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlpup.config import ConfigError, _check_keys, _load_yaml

_REQUIRED = {
    "model_config",
    "shard_index",
    "out_dir",
    "micro_batch_size",
    "grad_accum_steps",
    "max_lr",
    "warmup_steps",
    "total_steps",
    "decay_start_step",
    "eval_interval_steps",
    "eval_batches",
    "checkpoint_interval_steps",
    "log_interval_steps",
    "seed",
}
_OPTIONAL = {
    "eval_shard_index",
    "seq_len",
    "min_lr_ratio",
    "weight_decay",
    "beta1",
    "beta2",
    "grad_clip",
    "checkpoint_interval_minutes",
    "keep_checkpoints",
    "compile",
    "wandb_project",
    "wandb_run_name",
}


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Hyper-parameters for a single pretraining run.

    Self-validating: any construction (direct or via :func:`load_train_config`)
    enforces the schedule ordering ``warmup <= decay_start <= total`` and basic
    positivity, so an unrunnable schedule can never be built.
    """

    model_config: Path
    shard_index: Path
    out_dir: Path
    micro_batch_size: int
    grad_accum_steps: int
    max_lr: float
    warmup_steps: int
    total_steps: int
    decay_start_step: int
    eval_interval_steps: int
    eval_batches: int
    checkpoint_interval_steps: int
    log_interval_steps: int
    seed: int
    eval_shard_index: Path | None = None
    seq_len: int = 2048
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    checkpoint_interval_minutes: int = 30
    keep_checkpoints: int = 3
    compile: bool = False
    # Optional Weights & Biases tracking. ``wandb_project`` unset (None) means no
    # tracking and zero wandb imports; when set, the loop derives a run name from
    # ``out_dir`` + timestamp unless ``wandb_run_name`` overrides it.
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("micro_batch_size", self.micro_batch_size),
            ("grad_accum_steps", self.grad_accum_steps),
            ("seq_len", self.seq_len),
            ("total_steps", self.total_steps),
            ("keep_checkpoints", self.keep_checkpoints),
            ("eval_interval_steps", self.eval_interval_steps),
            ("eval_batches", self.eval_batches),
            ("checkpoint_interval_steps", self.checkpoint_interval_steps),
            ("log_interval_steps", self.log_interval_steps),
        ):
            if value < 1:
                raise ConfigError(f"{name} must be >= 1, got {value}")
        if self.max_lr <= 0:
            raise ConfigError(f"max_lr must be positive, got {self.max_lr}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ConfigError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        # grad_clip=0 reads as "no clipping" in some frameworks, but here it would
        # scale every gradient to zero and silently kill learning, so require > 0.
        if self.grad_clip <= 0:
            raise ConfigError(f"grad_clip must be positive, got {self.grad_clip}")
        if not 0.0 <= self.beta1 < 1.0:
            raise ConfigError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not 0.0 <= self.beta2 < 1.0:
            raise ConfigError(f"beta2 must be in [0, 1), got {self.beta2}")
        if self.weight_decay < 0:
            raise ConfigError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.checkpoint_interval_minutes <= 0:
            raise ConfigError(
                "checkpoint_interval_minutes must be positive, "
                f"got {self.checkpoint_interval_minutes}"
            )
        if not 0 <= self.warmup_steps <= self.decay_start_step <= self.total_steps:
            raise ConfigError(
                "schedule must satisfy 0 <= warmup_steps <= decay_start_step <= total_steps, "
                f"got warmup_steps={self.warmup_steps}, decay_start_step={self.decay_start_step}, "
                f"total_steps={self.total_steps}"
            )

    def effective_tokens_per_step(self, world_size: int = 1) -> int:
        """Tokens consumed per optimizer step across all ranks.

        Exposed as a method rather than a ``@property`` because ``world_size`` is
        a runtime quantity (from ``torchrun``), not a config field.
        """
        return self.micro_batch_size * self.grad_accum_steps * self.seq_len * world_size

    @property
    def min_lr(self) -> float:
        """Floor learning rate the decay tail settles at (``max_lr * ratio``)."""
        return self.max_lr * self.min_lr_ratio


def load_train_config(path: Path) -> TrainConfig:
    """Load and validate a training config from YAML."""
    raw = _load_yaml(path)
    _check_keys(raw, required=_REQUIRED, optional=_OPTIONAL, ctx=str(path))
    eval_index = raw.get("eval_shard_index")
    try:
        return TrainConfig(
            model_config=Path(str(raw["model_config"])),
            shard_index=Path(str(raw["shard_index"])),
            out_dir=Path(str(raw["out_dir"])),
            micro_batch_size=int(raw["micro_batch_size"]),
            grad_accum_steps=int(raw["grad_accum_steps"]),
            max_lr=float(raw["max_lr"]),
            warmup_steps=int(raw["warmup_steps"]),
            total_steps=int(raw["total_steps"]),
            decay_start_step=int(raw["decay_start_step"]),
            eval_interval_steps=int(raw["eval_interval_steps"]),
            eval_batches=int(raw["eval_batches"]),
            checkpoint_interval_steps=int(raw["checkpoint_interval_steps"]),
            log_interval_steps=int(raw["log_interval_steps"]),
            seed=int(raw["seed"]),
            eval_shard_index=None if eval_index is None else Path(str(eval_index)),
            seq_len=int(raw.get("seq_len", 2048)),
            min_lr_ratio=float(raw.get("min_lr_ratio", 0.1)),
            weight_decay=float(raw.get("weight_decay", 0.1)),
            beta1=float(raw.get("beta1", 0.9)),
            beta2=float(raw.get("beta2", 0.95)),
            grad_clip=float(raw.get("grad_clip", 1.0)),
            checkpoint_interval_minutes=int(raw.get("checkpoint_interval_minutes", 30)),
            keep_checkpoints=int(raw.get("keep_checkpoints", 3)),
            compile=bool(raw.get("compile", False)),
            wandb_project=(None if raw.get("wandb_project") is None else str(raw["wandb_project"])),
            wandb_run_name=(
                None if raw.get("wandb_run_name") is None else str(raw["wandb_run_name"])
            ),
        )
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
