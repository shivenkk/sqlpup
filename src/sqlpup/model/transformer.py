"""Llama-style decoder-only transformer (GQA, RoPE, SwiGLU, RMSNorm).

Deliberately conventional so checkpoints map 1:1 onto Hugging Face
``LlamaForCausalLM`` (no biases, pre-norm RMSNorm, SwiGLU MLP, rotary on q/k
only, tied embeddings). Tensor names differ from HF only by a ``model.``
prefix; :mod:`sqlpup.model.export` carries the mapping.

This module imports ``torch`` (the optional ``train`` extra) and is therefore
imported on demand, never at ``sqlpup`` package-import time.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from sqlpup.model.config import ModelConfig


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the last dim by half (HF/GPT-NeoX convention, not interleaved)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute rotary cos/sin tables of shape ``[seq_len, head_dim]``.

    Frequencies are computed in fp32 for numerical stability, then cast to
    ``dtype``. Matches HF Llama: ``inv_freq = 1 / theta**(arange(0,d,2)/d)`` and
    ``emb = cat(freqs, freqs)``.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)  # [seq_len, head_dim // 2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query and key tensors shaped ``[B, n_heads, T, head_dim]``."""
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads to match query heads (GQA), like HF ``repeat_kv``.

    ``n_rep`` is a static config-derived value, so the branch does not depend on
    runtime data and stays torch.compile-friendly.
    """
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm with Llama semantics (fp32 internal)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        h = x.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.eps)
        return self.weight * h.to(input_dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``. No biases."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Attention(nn.Module):
    """Grouped-query causal self-attention with RoPE. No biases."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_rep
        q_dim = config.n_heads * config.head_dim
        kv_dim = config.n_kv_heads * config.head_dim
        self.q_proj = nn.Linear(config.d_model, q_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        # is_causal=True selects the FlashAttention/efficient kernel on CUDA.
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return cast(Tensor, self.o_proj(attn))


class Block(nn.Module):
    """Pre-norm transformer block: attention then SwiGLU, both residual."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, config.norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class SqlpupLM(nn.Module):
    """Decoder-only text-to-SQL language model."""

    rope_cos: Tensor
    rope_sin: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {config.head_dim}")
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self._scale_residual_init()
        if config.tie_embeddings:
            # True storage sharing: the two modules reference one Parameter.
            self.lm_head.weight = self.embed_tokens.weight

        self._build_rope_cache(config.max_seq_len)

    # -- initialisation ----------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # RMSNorm weights stay at their 1.0 initialisation (Llama convention).

    def _scale_residual_init(self) -> None:
        # GPT-2/Llama residual scaling: shrink the projections that write back
        # into the residual stream so variance does not grow with depth.
        std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.layers:
            assert isinstance(block, Block)
            nn.init.normal_(block.self_attn.o_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=std)

    def _build_rope_cache(self, seq_len: int) -> None:
        # Non-persistent buffers: derived from rope_theta, kept out of the
        # state_dict so checkpoints map 1:1 onto HF Llama. Re-buildable -- a
        # later context-extension phase can call this with a larger seq_len (or
        # a patched theta) to resize/re-theta the rotary tables.
        cos, sin = build_rope_cache(seq_len, self.config.head_dim, self.config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # -- inference / analysis ---------------------------------------------

    def param_counts(self) -> tuple[int, int, int]:
        """Return ``(embedding, non_embedding, total)`` parameter counts.

        Tied embeddings are counted once (``parameters()`` deduplicates shared
        tensors). When untied, the separate output projection is vocab-scale and
        counted with the embeddings.
        """
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed_tokens.weight.numel()
        if self.lm_head.weight is not self.embed_tokens.weight:
            embedding += self.lm_head.weight.numel()
        return embedding, total - embedding, total

    def forward(self, tokens: Tensor) -> Tensor:
        _, t = tokens.shape
        if t > self.config.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.config.max_seq_len}")
        cos = self.rope_cos[:t]
        sin = self.rope_sin[:t]
        h = self.embed_tokens(tokens)
        for block in self.layers:
            h = block(h, cos, sin)
        h = self.norm(h)
        return cast(Tensor, self.lm_head(h))
