"""Export a sqlpup checkpoint to a HuggingFace ``LlamaForCausalLM`` directory.

The decoder in :mod:`sqlpup.model.transformer` is deliberately HF-Llama shaped
(RMSNorm pre-norm, SwiGLU, RoPE half-split, GQA, no biases, tied embeddings), so
the conversion is a pure key-rename plus a ``config.json`` payload -- no tensor
math. Our tensor names equal HF's minus the ``model.`` prefix; ``lm_head`` is
top-level in both.

Tied-embeddings convention (verified empirically, see the export report): for a
tied checkpoint we OMIT ``lm_head.weight`` from the weights file and set
``tie_word_embeddings: true`` so ``from_pretrained`` re-ties it from the input
embedding. This is both the HF convention and a hard requirement of the
safetensors format (it refuses to serialise two keys that share storage). A
stale/untied head is never emitted.

Like :mod:`sqlpup.model.transformer`, this module imports ``torch`` and is
therefore imported on demand (from the CLI command), never at ``sqlpup``
package-import time.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from sqlpup.model.config import ModelConfig

# ``<|end|>`` is sqlpup's document separator (the shard packer's EOS); it is the
# end-of-sequence token of the training scheme, so it maps to the HF eos. The
# scheme prepends no beginning-of-sequence token, so bos is always ``None``.
_EOS_TOKEN = "<|end|>"
_PAD_TOKEN = "<|pad|>"

_WEIGHTS_FILENAME = "model.safetensors"
_TOKENIZER_FILENAME = "tokenizer.json"
_TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
_CONFIG_FILENAME = "config.json"

# --dtype value -> (HF ``torch_dtype`` string, actual torch dtype for the weights).
_DTYPES: dict[str, tuple[str, torch.dtype]] = {
    "fp32": ("float32", torch.float32),
    "bf16": ("bfloat16", torch.bfloat16),
}


class ExportError(ValueError):
    """Raised when a checkpoint does not match the architecture it claims."""


def _expected_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """The exact sqlpup ``state_dict`` key -> shape map for ``cfg``.

    Includes ``lm_head.weight``: ``nn.Module.state_dict()`` emits it even when the
    embedding is tied (both names reference the one shared parameter).
    """
    d_model = cfg.d_model
    q_dim = cfg.n_heads * cfg.head_dim
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "embed_tokens.weight": (cfg.vocab_size, d_model),
        "norm.weight": (d_model,),
        "lm_head.weight": (cfg.vocab_size, d_model),
    }
    for i in range(cfg.n_layers):
        shapes[f"layers.{i}.input_layernorm.weight"] = (d_model,)
        shapes[f"layers.{i}.self_attn.q_proj.weight"] = (q_dim, d_model)
        shapes[f"layers.{i}.self_attn.k_proj.weight"] = (kv_dim, d_model)
        shapes[f"layers.{i}.self_attn.v_proj.weight"] = (kv_dim, d_model)
        shapes[f"layers.{i}.self_attn.o_proj.weight"] = (d_model, q_dim)
        shapes[f"layers.{i}.post_attention_layernorm.weight"] = (d_model,)
        shapes[f"layers.{i}.mlp.gate_proj.weight"] = (cfg.d_ff, d_model)
        shapes[f"layers.{i}.mlp.up_proj.weight"] = (cfg.d_ff, d_model)
        shapes[f"layers.{i}.mlp.down_proj.weight"] = (d_model, cfg.d_ff)
    return shapes


def to_hf_state_dict(state_dict: Mapping[str, Tensor], cfg: ModelConfig) -> dict[str, Tensor]:
    """Rename a sqlpup ``state_dict`` to HF ``LlamaForCausalLM`` keys.

    Strict: the input key set must be exactly the sqlpup set implied by ``cfg``
    and every tensor must have the expected shape -- unknown/missing keys and
    shape mismatches are errors, never silently dropped or passed through. For a
    tied config ``lm_head.weight`` is omitted from the output (HF re-ties it); it
    is kept, top-level, only when the config is untied.
    """
    expected = _expected_shapes(cfg)
    missing = expected.keys() - state_dict.keys()
    unexpected = state_dict.keys() - expected.keys()
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing keys {sorted(missing)}")
        if unexpected:
            detail.append(f"unexpected keys {sorted(unexpected)}")
        raise ExportError(
            "checkpoint state_dict does not match the sqlpup architecture: " + "; ".join(detail)
        )
    for key, shape in expected.items():
        got = tuple(state_dict[key].shape)
        if got != shape:
            raise ExportError(f"tensor {key!r} has shape {got}, expected {shape} for this config")

    tied = cfg.tie_embeddings
    if tied and not torch.equal(state_dict["lm_head.weight"], state_dict["embed_tokens.weight"]):
        raise ExportError(
            "config declares tie_embeddings=True but the checkpoint's lm_head.weight "
            "differs from embed_tokens.weight; refusing to emit a stale/untied head"
        )

    out: dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        if key == "lm_head.weight":
            if tied:
                continue  # omit; from_pretrained re-ties from the input embedding
            out["lm_head.weight"] = tensor  # untied: top-level, no `model.` prefix
        else:
            out[f"model.{key}"] = tensor
    return out


def to_hf_config(
    cfg: ModelConfig,
    *,
    torch_dtype: str = "float32",
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> dict[str, object]:
    """Build the ``config.json`` payload for a HF ``LlamaForCausalLM``."""
    return {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.d_ff,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": cfg.tie_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": torch_dtype,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_hf_directory(
    out_dir: Path,
    *,
    state_dict: Mapping[str, Tensor],
    cfg: ModelConfig,
    dtype: str = "fp32",
    tokenizer_path: Path | None = None,
) -> dict[str, object]:
    """Write a complete HF ``LlamaForCausalLM`` directory and return a summary.

    Emits ``config.json`` and ``model.safetensors``; when ``tokenizer_path`` is
    given it copies the ``tokenizer.json`` and writes a minimal
    ``tokenizer_config.json`` alongside. Weights are cast to ``dtype`` (``fp32``
    or ``bf16``).
    """
    if dtype not in _DTYPES:
        raise ExportError(f"unknown dtype {dtype!r}; expected one of {sorted(_DTYPES)}")
    hf_dtype_str, torch_dtype = _DTYPES[dtype]

    hf_state = {
        key: tensor.detach().to(torch_dtype).contiguous()
        for key, tensor in to_hf_state_dict(state_dict, cfg).items()
    }

    eos_token_id: int | None = None
    pad_present = False
    if tokenizer_path is not None:
        from sqlpup.tokenizer.train import load_tokenizer

        tokenizer = load_tokenizer(tokenizer_path)
        eos_token_id = tokenizer.token_to_id(_EOS_TOKEN)
        pad_present = tokenizer.token_to_id(_PAD_TOKEN) is not None

    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        out_dir / _CONFIG_FILENAME,
        to_hf_config(cfg, torch_dtype=hf_dtype_str, bos_token_id=None, eos_token_id=eos_token_id),
    )

    # safetensors is HF-standard, memory-mapped, and pickle-free; the tied lm_head
    # is already omitted, so no two keys share storage (which safetensors forbids).
    from safetensors.torch import save_file

    save_file(hf_state, str(out_dir / _WEIGHTS_FILENAME), metadata={"format": "pt"})

    tokenizer_written = False
    if tokenizer_path is not None:
        shutil.copyfile(tokenizer_path, out_dir / _TOKENIZER_FILENAME)
        _write_json(
            out_dir / _TOKENIZER_CONFIG_FILENAME,
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": cfg.max_seq_len,
                "bos_token": None,
                "eos_token": _EOS_TOKEN if eos_token_id is not None else None,
                "pad_token": _PAD_TOKEN if pad_present else None,
            },
        )
        tokenizer_written = True

    return {
        "out": str(out_dir),
        "weights": _WEIGHTS_FILENAME,
        "tensors": len(hf_state),
        "dtype": hf_dtype_str,
        "tie_word_embeddings": cfg.tie_embeddings,
        "tokenizer": tokenizer_written,
        "eos_token_id": eos_token_id,
    }
