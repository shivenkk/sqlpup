"""Unit tests for the sweep-config generator's importable core (torch-free).

Loads ``scripts/make_sweep_configs.py`` the same way ``test_smoke_train`` loads
the smoke script, exercises the pure LR-variant derivation, and drives ``main``
end-to-end -- asserting every generated YAML loads back as a valid
:class:`~sqlpup.train.config.TrainConfig`, i.e. the sweep configs are runnable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from sqlpup.train.config import load_train_config

BASE_YAML = """\
model_config: configs/model/proxy_30m.yaml
shard_index: artifacts/shards/train/index.json
eval_shard_index: artifacts/shards/eval/index.json
out_dir: artifacts/checkpoints/proxy_30m
seq_len: 2048
micro_batch_size: 16
grad_accum_steps: 4
max_lr: 3.0e-3
min_lr_ratio: 0.1
warmup_steps: 40
decay_start_step: 1600
total_steps: 2000
eval_interval_steps: 200
eval_batches: 50
checkpoint_interval_steps: 500
log_interval_steps: 10
keep_checkpoints: 3
seed: 1337
wandb_project: sqlpup-proxy
"""


def _load_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_sweep_configs.py"
    spec = importlib.util.spec_from_file_location("make_sweep_configs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_module()


def _base() -> dict[str, Any]:
    parsed: dict[str, Any] = yaml.safe_load(BASE_YAML)
    return parsed


def test_variant_name_and_filename() -> None:
    assert sweep.variant_name("proxy_30m", "1.5e-3") == "proxy_30m-lr1.5e-3"
    assert sweep.variant_filename("proxy_30m", "6e-3") == "proxy_30m-lr6e-3.yaml"


def test_derive_variant_sets_lr_out_dir_and_run_name() -> None:
    base = _base()
    variant = sweep.derive_variant(base, "proxy_30m", "1.5e-3")
    # max_lr is the numeric value of the tag; the tag string drives the names.
    assert variant["max_lr"] == pytest.approx(1.5e-3)
    assert variant["out_dir"] == "artifacts/checkpoints/proxy_30m-lr1.5e-3"
    assert variant["wandb_run_name"] == "proxy_30m-lr1.5e-3"
    # Every other field is copied verbatim.
    for key in base:
        if key not in {"max_lr", "out_dir"}:
            assert variant[key] == base[key]
    # The derivation does not mutate the base mapping.
    assert base["max_lr"] == pytest.approx(3.0e-3)
    assert "wandb_run_name" not in base


def test_derive_variant_without_wandb_project_adds_no_run_name() -> None:
    base = _base()
    del base["wandb_project"]
    variant = sweep.derive_variant(base, "proxy_30m", "3e-3")
    assert "wandb_run_name" not in variant  # only set when tracking is configured
    assert variant["max_lr"] == pytest.approx(3e-3)
    assert variant["out_dir"] == "artifacts/checkpoints/proxy_30m-lr3e-3"


def test_main_writes_one_valid_config_per_lr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_path = tmp_path / "proxy_30m.yaml"
    base_path.write_text(BASE_YAML, encoding="utf-8")
    out_dir = tmp_path / "sweep"

    code = sweep.main(
        [
            "--config",
            str(base_path),
            "--lrs",
            "1.5e-3",
            "3e-3",
            "6e-3",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert len(summary["written"]) == 3

    expected = {
        "1.5e-3": 1.5e-3,
        "3e-3": 3e-3,
        "6e-3": 6e-3,
    }
    for tag, lr in expected.items():
        path = out_dir / f"proxy_30m-lr{tag}.yaml"
        assert path.exists()
        # The generated config loads and validates as a runnable TrainConfig.
        cfg = load_train_config(path)
        assert cfg.max_lr == pytest.approx(lr)
        assert cfg.out_dir == Path(f"artifacts/checkpoints/proxy_30m-lr{tag}")
        assert cfg.wandb_run_name == f"proxy_30m-lr{tag}"
        # Untouched fields survive the round-trip.
        assert cfg.total_steps == 2000
        assert cfg.seed == 1337
        assert cfg.wandb_project == "sqlpup-proxy"
