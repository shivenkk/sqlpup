from pathlib import Path

import pytest

from sqlpup.config import ConfigError
from sqlpup.model.config import ModelConfig, load_model_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "model"


def test_shipped_base_config_loads() -> None:
    cfg = load_model_config(CONFIG_DIR / "base_260m.yaml")
    assert cfg.d_model == 1024
    assert cfg.n_layers == 20
    assert cfg.n_heads == 16
    assert cfg.n_kv_heads == 4
    assert cfg.d_ff == 2816
    assert cfg.vocab_size == 32768
    assert cfg.max_seq_len == 2048
    assert cfg.tie_embeddings is True
    assert cfg.head_dim == 64


def test_shipped_base_400m_config_loads() -> None:
    cfg = load_model_config(CONFIG_DIR / "base_400m.yaml")
    assert cfg.d_model == 1024
    assert cfg.n_layers == 32
    assert cfg.n_heads == 16
    assert cfg.n_kv_heads == 4
    assert cfg.d_ff == 2816
    assert cfg.vocab_size == 32768
    assert cfg.max_seq_len == 2048
    assert cfg.tie_embeddings is True
    assert cfg.head_dim == 64
    assert cfg.n_heads == 4 * cfg.n_kv_heads  # GQA 4:1 preserved


@pytest.mark.parametrize("name", ["proxy_30m", "proxy_50m"])
def test_shipped_proxy_configs_load(name: str) -> None:
    cfg = load_model_config(CONFIG_DIR / f"{name}.yaml")
    # same GQA family: 4:1 query:kv ratio, head_dim a multiple of 8 and >= 32.
    assert cfg.n_heads == 4 * cfg.n_kv_heads
    assert cfg.head_dim >= 32
    assert cfg.head_dim % 8 == 0
    assert cfg.vocab_size == 32768


def test_defaults_applied_when_optional_keys_omitted(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("d_model: 64\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nd_ff: 128\n")
    cfg = load_model_config(path)
    assert cfg.vocab_size == 32768
    assert cfg.max_seq_len == 2048
    assert cfg.rope_theta == 10000.0
    assert cfg.norm_eps == 1e-5
    assert cfg.tie_embeddings is True


def test_roundtrip_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "d_model: 128\nn_layers: 3\nn_heads: 8\nn_kv_heads: 2\nd_ff: 256\n"
        "vocab_size: 1000\nmax_seq_len: 512\nrope_theta: 50000.0\n"
        "norm_eps: 0.0001\ntie_embeddings: false\n"
    )
    cfg = load_model_config(path)
    assert cfg == ModelConfig(
        d_model=128,
        n_layers=3,
        n_heads=8,
        n_kv_heads=2,
        d_ff=256,
        vocab_size=1000,
        max_seq_len=512,
        rope_theta=50000.0,
        norm_eps=0.0001,
        tie_embeddings=False,
    )


def test_d_model_not_divisible_by_n_heads_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("d_model: 100\nn_layers: 2\nn_heads: 8\nn_kv_heads: 2\nd_ff: 128\n")
    with pytest.raises(ConfigError, match="divisible by n_heads"):
        load_model_config(path)


def test_n_heads_not_divisible_by_n_kv_heads_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("d_model: 64\nn_layers: 2\nn_heads: 8\nn_kv_heads: 3\nd_ff: 128\n")
    with pytest.raises(ConfigError, match="divisible by n_kv_heads"):
        load_model_config(path)


def test_non_positive_dim_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("d_model: 0\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nd_ff: 128\n")
    with pytest.raises(ConfigError, match="must be positive"):
        load_model_config(path)


def test_non_positive_vocab_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "d_model: 64\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nd_ff: 128\nvocab_size: 0\n"
    )
    with pytest.raises(ConfigError, match="must be positive"):
        load_model_config(path)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("d_model: 64\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nd_ff: 128\nfoo: 1\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_model_config(path)


def test_missing_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("d_model: 64\nn_layers: 2\n")
    with pytest.raises(ConfigError, match="missing required keys"):
        load_model_config(path)
