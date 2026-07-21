"""Tests for the HF-Llama exporter (``sqlpup.model.export``) and ``sqlpup export``.

Everything here runs offline on tiny configs. The pure key-rename / config-payload
tests need only torch; the tests that write an export directory additionally need
``safetensors``; the equivalence proof additionally needs ``transformers`` and is
skipped cleanly when it is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from sqlpup.cli import main
from sqlpup.model.config import ModelConfig
from sqlpup.model.export import (
    ExportError,
    to_hf_config,
    to_hf_state_dict,
    write_hf_directory,
)
from sqlpup.model.transformer import SqlpupLM


def tiny_config(**overrides: object) -> ModelConfig:
    params: dict[str, object] = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 128,
        "vocab_size": 128,
        "max_seq_len": 64,
    }
    params.update(overrides)
    return ModelConfig(**params)  # type: ignore[arg-type]


def _per_layer_hf_keys(i: int) -> set[str]:
    return {
        f"model.layers.{i}.input_layernorm.weight",
        f"model.layers.{i}.self_attn.q_proj.weight",
        f"model.layers.{i}.self_attn.k_proj.weight",
        f"model.layers.{i}.self_attn.v_proj.weight",
        f"model.layers.{i}.self_attn.o_proj.weight",
        f"model.layers.{i}.post_attention_layernorm.weight",
        f"model.layers.{i}.mlp.gate_proj.weight",
        f"model.layers.{i}.mlp.up_proj.weight",
        f"model.layers.{i}.mlp.down_proj.weight",
    }


# --- to_hf_state_dict: key rename + strictness ----------------------------


def test_state_dict_exact_hf_key_set_tied() -> None:
    cfg = tiny_config()  # tied by default
    model = SqlpupLM(cfg)
    hf = to_hf_state_dict(model.state_dict(), cfg)

    expected = {"model.embed_tokens.weight", "model.norm.weight"}
    expected |= _per_layer_hf_keys(0) | _per_layer_hf_keys(1)
    # Tied: lm_head is OMITTED so from_pretrained re-ties (never emit a stale copy).
    assert set(hf) == expected
    assert "lm_head.weight" not in hf


def test_state_dict_shapes_are_preserved() -> None:
    cfg = tiny_config()
    sd = SqlpupLM(cfg).state_dict()
    hf = to_hf_state_dict(sd, cfg)
    # Every renamed tensor keeps the exact source tensor (identity, not a copy).
    assert hf["model.embed_tokens.weight"] is sd["embed_tokens.weight"]
    assert hf["model.layers.0.self_attn.q_proj.weight"].shape == (64, 64)
    assert hf["model.layers.0.self_attn.k_proj.weight"].shape == (32, 64)
    assert hf["model.layers.0.mlp.gate_proj.weight"].shape == (128, 64)
    assert hf["model.norm.weight"].shape == (64,)


def test_state_dict_untied_keeps_top_level_lm_head() -> None:
    cfg = tiny_config(tie_embeddings=False)
    model = SqlpupLM(cfg)
    hf = to_hf_state_dict(model.state_dict(), cfg)
    # Untied: lm_head stays, top-level (no `model.` prefix), matching HF.
    assert "lm_head.weight" in hf
    assert "model.lm_head.weight" not in hf
    assert hf["lm_head.weight"].shape == (128, 64)


def test_state_dict_rejects_extra_key() -> None:
    cfg = tiny_config()
    sd = dict(SqlpupLM(cfg).state_dict())
    sd["layers.0.self_attn.rotary_emb.inv_freq"] = torch.zeros(32)
    with pytest.raises(ExportError, match="unexpected"):
        to_hf_state_dict(sd, cfg)


def test_state_dict_rejects_missing_key() -> None:
    cfg = tiny_config()
    sd = dict(SqlpupLM(cfg).state_dict())
    del sd["norm.weight"]
    with pytest.raises(ExportError, match="missing"):
        to_hf_state_dict(sd, cfg)


def test_state_dict_rejects_wrong_shape() -> None:
    cfg = tiny_config()
    sd = dict(SqlpupLM(cfg).state_dict())
    sd["norm.weight"] = torch.zeros(63)  # should be (64,)
    with pytest.raises(ExportError, match="shape"):
        to_hf_state_dict(sd, cfg)


def test_state_dict_rejects_stale_tied_lm_head() -> None:
    # Config says tied, but the checkpoint's lm_head diverges from the embedding:
    # emitting it (or trusting the tie) would ship a stale head, so it must error.
    cfg = tiny_config()
    sd = dict(SqlpupLM(cfg).state_dict())
    sd["lm_head.weight"] = torch.randn(128, 64)
    with pytest.raises(ExportError, match="tie"):
        to_hf_state_dict(sd, cfg)


# --- to_hf_config: golden payload -----------------------------------------


def test_config_golden() -> None:
    cfg = tiny_config()
    payload = to_hf_config(cfg, torch_dtype="float32", bos_token_id=None, eos_token_id=7)
    assert payload == {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "bos_token_id": None,
        "eos_token_id": 7,
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 128,
        "max_position_embeddings": 64,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "vocab_size": 128,
    }


def test_config_reflects_dtype_and_tie() -> None:
    cfg = tiny_config(tie_embeddings=False, rope_theta=500000.0)
    payload = to_hf_config(cfg, torch_dtype="bfloat16", eos_token_id=3)
    assert payload["torch_dtype"] == "bfloat16"
    assert payload["tie_word_embeddings"] is False
    assert payload["rope_theta"] == 500000.0
    assert payload["eos_token_id"] == 3
    assert payload["bos_token_id"] is None


# --- write_hf_directory + CLI end-to-end ----------------------------------


def _make_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models

    tok = Tokenizer(models.BPE())
    tok.add_special_tokens(["<|end|>", "<|pad|>"])
    tok.save(str(path))


def _write_checkpoint(path: Path, model: SqlpupLM, model_config_path: Path) -> None:
    payload = {
        "step": 0,
        "model": model.state_dict(),
        "train_config": {"model_config": str(model_config_path)},
    }
    torch.save(payload, path)


def _write_model_config(path: Path, cfg: ModelConfig) -> None:
    path.write_text(
        f"d_model: {cfg.d_model}\n"
        f"n_layers: {cfg.n_layers}\n"
        f"n_heads: {cfg.n_heads}\n"
        f"n_kv_heads: {cfg.n_kv_heads}\n"
        f"d_ff: {cfg.d_ff}\n"
        f"vocab_size: {cfg.vocab_size}\n"
        f"max_seq_len: {cfg.max_seq_len}\n",
        encoding="utf-8",
    )


def test_write_directory_files_and_reload(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    cfg = tiny_config()
    model = SqlpupLM(cfg)
    out = tmp_path / "hf"
    write_hf_directory(out, state_dict=model.state_dict(), cfg=cfg, dtype="fp32")

    assert (out / "config.json").is_file()
    assert (out / "model.safetensors").is_file()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["torch_dtype"] == "float32"

    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        assert "lm_head.weight" not in keys  # tied
        assert "model.embed_tokens.weight" in keys
        embed = f.get_tensor("model.embed_tokens.weight")
        assert embed.dtype == torch.float32
        assert embed.shape == (128, 64)


def test_write_directory_bf16(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    cfg = tiny_config()
    out = tmp_path / "hf"
    write_hf_directory(out, state_dict=SqlpupLM(cfg).state_dict(), cfg=cfg, dtype="bf16")

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["torch_dtype"] == "bfloat16"
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        assert f.get_tensor("model.norm.weight").dtype == torch.bfloat16


def test_write_directory_with_tokenizer(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")

    cfg = tiny_config()
    tok_path = tmp_path / "tokenizer.json"
    _make_tokenizer(tok_path)
    out = tmp_path / "hf"
    write_hf_directory(
        out, state_dict=SqlpupLM(cfg).state_dict(), cfg=cfg, dtype="fp32", tokenizer_path=tok_path
    )

    assert (out / "tokenizer.json").is_file()
    tok_config = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tok_config["tokenizer_class"] == "PreTrainedTokenizerFast"
    # <|end|> is the sqlpup document separator -> the HF eos.
    from sqlpup.tokenizer.train import load_tokenizer

    end_id = load_tokenizer(tok_path).token_to_id("<|end|>")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["eos_token_id"] == end_id
    assert config["bos_token_id"] is None


def test_cli_export_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    cfg = tiny_config()
    model = SqlpupLM(cfg)
    model_config_path = tmp_path / "model.yaml"
    _write_model_config(model_config_path, cfg)
    ckpt = tmp_path / "step_0.pt"
    _write_checkpoint(ckpt, model, model_config_path)
    out = tmp_path / "export"

    rc = main(["export", "--checkpoint", str(ckpt), "--out", str(out)])
    assert rc == 0
    assert (out / "config.json").is_file()
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        assert f.get_tensor("model.embed_tokens.weight").shape == (128, 64)


# --- the equivalence proof (guarded) --------------------------------------


def test_export_matches_hf_llama_tiny(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(0)
    cfg = tiny_config()
    model = SqlpupLM(cfg).eval()
    out = tmp_path / "hf"
    write_hf_directory(out, state_dict=model.state_dict(), cfg=cfg, dtype="fp32")

    hf, info = transformers.LlamaForCausalLM.from_pretrained(
        out, local_files_only=True, torch_dtype=torch.float32, output_loading_info=True
    )
    hf.eval()
    # Clean round-trip: nothing unexpected/mismatched; the only "missing" key
    # allowed is the tied lm_head, which HF re-ties from the embedding. (The HF
    # loading-info collections are set-or-list depending on the version.)
    assert not info["unexpected_keys"]
    assert not info["mismatched_keys"]
    assert set(info["missing_keys"]) <= {"lm_head.weight"}
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()

    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        ours = model(tokens)
        theirs = hf(tokens).logits
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)
