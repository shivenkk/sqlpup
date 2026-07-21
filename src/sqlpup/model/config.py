"""Typed configuration for the model architecture.

Mirrors the loader conventions in :mod:`sqlpup.config` (frozen slotted
dataclass, strict YAML parsing with unknown/missing-key rejection). The loader
lives beside :class:`ModelConfig` rather than in ``sqlpup.config`` so that all
model-shape concerns stay in the ``sqlpup.model`` package, while the shared
low-level helpers (``_load_yaml``, ``_check_keys``, :class:`ConfigError`) are
reused to keep parsing/validation behaviour identical across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlpup.config import ConfigError, _check_keys, _load_yaml


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture hyper-parameters for the Llama-style GQA decoder.

    Self-validating: any construction (direct or via :func:`load_model_config`)
    enforces the dimensional invariants, so an invalid model can never exist.
    """

    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int = 32768
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("n_kv_heads", self.n_kv_heads),
            ("d_ff", self.d_ff),
            ("vocab_size", self.vocab_size),
            ("max_seq_len", self.max_seq_len),
        ):
            if value <= 0:
                raise ConfigError(f"{name} must be positive, got {value}")
        if self.d_model % self.n_heads != 0:
            raise ConfigError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ConfigError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Per-head dimension (``d_model / n_heads``)."""
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share each KV head (``n_heads / n_kv_heads``)."""
        return self.n_heads // self.n_kv_heads


def load_model_config(path: Path) -> ModelConfig:
    """Load and validate a model architecture config from YAML."""
    raw = _load_yaml(path)
    _check_keys(
        raw,
        required={"d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff"},
        optional={"vocab_size", "max_seq_len", "rope_theta", "norm_eps", "tie_embeddings"},
        ctx=str(path),
    )
    try:
        return ModelConfig(
            d_model=int(raw["d_model"]),
            n_layers=int(raw["n_layers"]),
            n_heads=int(raw["n_heads"]),
            n_kv_heads=int(raw["n_kv_heads"]),
            d_ff=int(raw["d_ff"]),
            vocab_size=int(raw.get("vocab_size", 32768)),
            max_seq_len=int(raw.get("max_seq_len", 2048)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            norm_eps=float(raw.get("norm_eps", 1e-5)),
            tie_embeddings=bool(raw.get("tie_embeddings", True)),
        )
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
