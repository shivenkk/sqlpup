from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("torch")

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from sqlpup.model.config import ModelConfig, load_model_config
from sqlpup.model.transformer import (
    Block,
    RMSNorm,
    SqlpupLM,
    apply_rotary_emb,
    build_rope_cache,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "model"


def tiny_config(**overrides: object) -> ModelConfig:
    params: dict[str, object] = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 128,
        "vocab_size": 128,
        "max_seq_len": 32,
    }
    params.update(overrides)
    return ModelConfig(**params)  # type: ignore[arg-type]


# --- shape / bounds -------------------------------------------------------


def test_forward_shape() -> None:
    model = SqlpupLM(tiny_config(max_seq_len=32))
    tokens = torch.randint(0, 128, (3, 20))
    out = model(tokens)
    assert out.shape == (3, 20, 128)


def test_forward_rejects_too_long_sequence() -> None:
    model = SqlpupLM(tiny_config(max_seq_len=16))
    tokens = torch.randint(0, 128, (1, 17))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(tokens)


# --- causality ------------------------------------------------------------


def test_causality_exact() -> None:
    torch.manual_seed(0)
    model = SqlpupLM(tiny_config()).eval()
    tokens = torch.randint(0, 128, (2, 16))
    j = 10
    with torch.no_grad():
        logits1 = model(tokens)
        perturbed = tokens.clone()
        perturbed[:, j] = (perturbed[:, j] + 1) % 128
        logits2 = model(perturbed)
    # A change at position j cannot affect outputs at positions < j.
    torch.testing.assert_close(logits1[:, :j], logits2[:, :j], rtol=0, atol=0)
    # ...but must affect position j onward.
    assert not torch.equal(logits1[:, j:], logits2[:, j:])


# --- GQA ------------------------------------------------------------------


def test_gqa_projection_shapes_and_forward() -> None:
    cfg = tiny_config(n_heads=4, n_kv_heads=1)
    model = SqlpupLM(cfg)
    attn = cast(Block, model.layers[0]).self_attn
    hd = cfg.head_dim
    assert attn.q_proj.weight.shape == (cfg.n_heads * hd, cfg.d_model)
    assert attn.k_proj.weight.shape == (cfg.n_kv_heads * hd, cfg.d_model)
    assert attn.v_proj.weight.shape == (cfg.n_kv_heads * hd, cfg.d_model)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    assert model(tokens).shape == (1, 8, cfg.vocab_size)


def test_mha_degenerate_case() -> None:
    cfg = tiny_config(n_heads=4, n_kv_heads=4)
    model = SqlpupLM(cfg)
    attn = cast(Block, model.layers[0]).self_attn
    assert attn.k_proj.weight.shape == (cfg.d_model, cfg.d_model)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    assert model(tokens).shape == (1, 8, cfg.vocab_size)


# --- weight tying ---------------------------------------------------------


def test_embeddings_tied_by_default() -> None:
    model = SqlpupLM(tiny_config())
    assert model.lm_head.weight is model.embed_tokens.weight
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()


def test_embeddings_untied_when_disabled() -> None:
    model = SqlpupLM(tiny_config(tie_embeddings=False))
    assert model.lm_head.weight is not model.embed_tokens.weight
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()


# --- no biases ------------------------------------------------------------


def test_no_biases_anywhere() -> None:
    model = SqlpupLM(tiny_config())
    for name, _ in model.named_parameters():
        assert not name.endswith("bias"), name
    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None


# --- RMSNorm fp32 semantics ----------------------------------------------


def test_rmsnorm_unit_rms_and_dtype() -> None:
    norm = RMSNorm(64, eps=1e-6)  # weight initialised to ones
    x = torch.randn(8, 64) * 5.0
    out = norm(x)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out.pow(2).mean(-1), torch.ones(8), rtol=1e-3, atol=1e-3)


def test_rmsnorm_computes_in_fp32_for_low_precision_input() -> None:
    # A bf16 model (weights + activations bf16), as in real training.
    torch.manual_seed(0)
    norm = RMSNorm(32, eps=1e-6).to(torch.bfloat16)
    nn.init.normal_(norm.weight)
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    out = norm(x)
    assert out.dtype == torch.bfloat16  # cast back to the input dtype
    # Reference does the normalisation in fp32 (Llama semantics); a naive
    # all-bf16 implementation would compute the variance imprecisely and differ.
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = norm.weight * normed.to(torch.bfloat16)
    torch.testing.assert_close(out, ref)


# --- RoPE relative-position property -------------------------------------


def test_rope_encodes_relative_position() -> None:
    # The defining RoPE property: with position-independent content, the QK
    # score between positions i and j depends only on the offset (i - j).
    torch.manual_seed(0)
    head_dim, seq_len = 8, 16
    cos, sin = build_rope_cache(seq_len, head_dim, 10000.0, dtype=torch.float64)
    q0 = torch.randn(head_dim, dtype=torch.float64)
    k0 = torch.randn(head_dim, dtype=torch.float64)
    q = q0.view(1, 1, 1, head_dim).expand(1, 1, seq_len, head_dim).contiguous()
    k = k0.view(1, 1, 1, head_dim).expand(1, 1, seq_len, head_dim).contiguous()
    qr, kr = apply_rotary_emb(q, k, cos, sin)

    def score(i: int, j: int) -> torch.Tensor:
        return torch.dot(qr[0, 0, i], kr[0, 0, j])

    torch.testing.assert_close(score(2, 5), score(4, 7))  # offset -3
    torch.testing.assert_close(score(1, 6), score(9, 14))  # offset -5
    assert not torch.isclose(score(2, 5), score(2, 6))  # different offset differs


def test_rope_matches_hf_half_split_convention() -> None:
    # Lock RoPE to the HF/GPT-NeoX half-split convention: it is load-bearing for
    # the 1:1 HF/GGUF/vLLM export story, yet the relative-position property above
    # would pass equally for an interleaved implementation. The expected rotation
    # is computed here from first principles (the HF formula) using no module
    # rope helper, so a convention swap is genuinely caught.
    head_dim, theta, position = 8, 10000.0, 3

    # Independent HF formula: inv_freq = 1 / theta**(2i/d); angles = position *
    # inv_freq; the rotary cos/sin rows are cos/sin of cat(angles, angles).
    i = torch.arange(0, head_dim // 2, dtype=torch.float64)
    inv_freq = 1.0 / (theta ** (2 * i / head_dim))
    angles = position * inv_freq
    cos_row = torch.cat((angles, angles)).cos()
    sin_row = torch.cat((angles, angles)).sin()

    # A fixed asymmetric input whose two halves differ, chosen so the half-split
    # and interleaved conventions genuinely disagree (asserted below).
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)

    # Half-split rotate_half(x) = cat(-x2, x1) with x1 the first half.
    x1, x2 = x[: head_dim // 2], x[head_dim // 2 :]
    expected = x * cos_row + torch.cat((-x2, x1)) * sin_row

    # The interleaved convention rotates adjacent pairs with cos/sin laid out as
    # [c0, c0, c1, c1, ...]; build it and confirm it differs on this input, so
    # the test cannot pass under both conventions.
    cos_il = angles.cos().repeat_interleave(2)
    sin_il = angles.sin().repeat_interleave(2)
    pairs = x.view(head_dim // 2, 2)
    rotate_interleaved = torch.stack((-pairs[:, 1], pairs[:, 0]), dim=1).reshape(head_dim)
    interleaved = x * cos_il + rotate_interleaved * sin_il
    assert not torch.allclose(expected, interleaved)  # the two conventions disagree here

    # The model's actual rope-application path, fed cos/sin for positions 0..position.
    cos, sin = build_rope_cache(position + 1, head_dim, theta, dtype=torch.float64)
    q = x.view(1, 1, 1, head_dim).expand(1, 1, position + 1, head_dim).contiguous()
    q_rot, k_rot = apply_rotary_emb(q, q.clone(), cos, sin)

    torch.testing.assert_close(q_rot[0, 0, position], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k_rot[0, 0, position], expected, rtol=1e-5, atol=1e-6)


# --- gradient flow --------------------------------------------------------


def test_gradient_flow_reaches_every_parameter() -> None:
    torch.manual_seed(0)
    model = SqlpupLM(tiny_config())
    tokens = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    logits = model(tokens)
    loss = F.cross_entropy(logits.view(-1, 128), targets.view(-1))
    loss.backward()  # type: ignore[no-untyped-call]
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


# --- determinism ----------------------------------------------------------


def test_determinism_same_seed_same_logits() -> None:
    cfg = tiny_config()
    tokens = torch.randint(0, 128, (2, 8))
    torch.manual_seed(123)
    m1 = SqlpupLM(cfg)
    torch.manual_seed(123)
    m2 = SqlpupLM(cfg)
    with torch.no_grad():
        out1 = m1(tokens)
        out2 = m2(tokens)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


# --- torch.compile smoke --------------------------------------------------


def test_torch_compile_smoke() -> None:
    torch.manual_seed(0)
    model = SqlpupLM(tiny_config()).eval()
    tokens = torch.randint(0, 128, (1, 8))
    try:
        compiled = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            out = compiled(tokens)
    except Exception as exc:  # pragma: no cover - depends on CI toolchain
        pytest.skip(f"torch.compile unavailable: {exc}")
    assert out.shape == (1, 8, 128)


# --- parameter counts (formula computed independently here) ---------------


def test_base_260m_param_count_matches_formula() -> None:
    cfg = load_model_config(CONFIG_DIR / "base_260m.yaml")
    with torch.device("meta"):
        model = SqlpupLM(cfg)
    embedding, non_embedding, total = model.param_counts()

    head_dim = cfg.d_model // cfg.n_heads
    attn = 2 * cfg.d_model * cfg.d_model + 2 * cfg.d_model * (cfg.n_kv_heads * head_dim)
    mlp = 3 * cfg.d_model * cfg.d_ff
    norms = 2 * cfg.d_model
    per_layer = attn + mlp + norms
    expected_embedding = cfg.vocab_size * cfg.d_model
    expected_non_embedding = cfg.n_layers * per_layer + cfg.d_model  # + final norm
    expected_total = expected_embedding + expected_non_embedding  # tied head adds 0

    assert embedding == expected_embedding
    assert non_embedding == expected_non_embedding
    assert total == expected_total
    assert 259_000_000 <= total <= 261_000_000
    assert abs(non_embedding - 225_000_000) <= 0.02 * 225_000_000


def test_base_400m_param_count_matches_formula() -> None:
    cfg = load_model_config(CONFIG_DIR / "base_400m.yaml")
    with torch.device("meta"):
        model = SqlpupLM(cfg)
    embedding, non_embedding, total = model.param_counts()

    head_dim = cfg.d_model // cfg.n_heads
    attn = 2 * cfg.d_model * cfg.d_model + 2 * cfg.d_model * (cfg.n_kv_heads * head_dim)
    mlp = 3 * cfg.d_model * cfg.d_ff
    norms = 2 * cfg.d_model
    per_layer = attn + mlp + norms
    expected_embedding = cfg.vocab_size * cfg.d_model
    expected_non_embedding = cfg.n_layers * per_layer + cfg.d_model  # + final norm
    expected_total = expected_embedding + expected_non_embedding  # tied head adds 0

    assert embedding == expected_embedding
    assert non_embedding == expected_non_embedding
    assert total == expected_total
    # P2 scale-decision target window, plus the exact numbers computed for this config.
    assert 390_000_000 <= total <= 410_000_000
    assert total == 394_331_136
    assert non_embedding == 360_776_704
    assert head_dim == 64  # head_dim locked at 64 (GQA 4:1, RoPE) across all scales


@pytest.mark.parametrize(("name", "target"), [("proxy_30m", 30_000_000), ("proxy_50m", 50_000_000)])
def test_proxy_configs_hit_param_targets(name: str, target: int) -> None:
    cfg = load_model_config(CONFIG_DIR / f"{name}.yaml")
    with torch.device("meta"):
        model = SqlpupLM(cfg)
    _, _, total = model.param_counts()
    assert abs(total - target) <= 0.10 * target
