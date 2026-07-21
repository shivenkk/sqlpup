import dataclasses
from pathlib import Path

import pytest

from sqlpup.config import ConfigError
from sqlpup.train.config import TrainConfig, load_train_config

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_DIR = REPO_ROOT / "configs" / "train"

REQUIRED_YAML = """
model_config: configs/model/proxy_30m.yaml
shard_index: artifacts/shards/index.json
out_dir: artifacts/run
micro_batch_size: 4
grad_accum_steps: 8
max_lr: 0.0006
warmup_steps: 100
total_steps: 1000
decay_start_step: 800
eval_interval_steps: 200
eval_batches: 20
checkpoint_interval_steps: 500
log_interval_steps: 10
seed: 1337
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(body)
    return path


def test_loads_required_fields(tmp_path: Path) -> None:
    cfg = load_train_config(_write(tmp_path, REQUIRED_YAML))
    assert cfg.model_config == Path("configs/model/proxy_30m.yaml")
    assert cfg.shard_index == Path("artifacts/shards/index.json")
    assert cfg.out_dir == Path("artifacts/run")
    assert cfg.micro_batch_size == 4
    assert cfg.grad_accum_steps == 8
    assert cfg.max_lr == pytest.approx(0.0006)
    assert cfg.warmup_steps == 100
    assert cfg.total_steps == 1000
    assert cfg.decay_start_step == 800
    assert cfg.seed == 1337


def test_defaults_applied(tmp_path: Path) -> None:
    cfg = load_train_config(_write(tmp_path, REQUIRED_YAML))
    assert cfg.eval_shard_index is None
    assert cfg.seq_len == 2048
    assert cfg.min_lr_ratio == pytest.approx(0.1)
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.beta1 == pytest.approx(0.9)
    assert cfg.beta2 == pytest.approx(0.95)
    assert cfg.grad_clip == pytest.approx(1.0)
    assert cfg.checkpoint_interval_minutes == 30
    assert cfg.keep_checkpoints == 3
    assert cfg.compile is False
    assert cfg.wandb_project is None  # tracking off by default
    assert cfg.wandb_run_name is None


def test_optional_fields_overridable(tmp_path: Path) -> None:
    body = (
        REQUIRED_YAML + "eval_shard_index: artifacts/eval/index.json\nseq_len: 512\ncompile: true\n"
    )
    cfg = load_train_config(_write(tmp_path, body))
    assert cfg.eval_shard_index == Path("artifacts/eval/index.json")
    assert cfg.seq_len == 512
    assert cfg.compile is True


def test_wandb_fields_load(tmp_path: Path) -> None:
    body = REQUIRED_YAML + "wandb_project: sqlpup-proxy\nwandb_run_name: proxy_30m-lr3e-3\n"
    cfg = load_train_config(_write(tmp_path, body))
    assert cfg.wandb_project == "sqlpup-proxy"
    assert cfg.wandb_run_name == "proxy_30m-lr3e-3"


def test_effective_tokens_per_step() -> None:
    cfg = TrainConfig(
        model_config=Path("m.yaml"),
        shard_index=Path("i.json"),
        out_dir=Path("out"),
        micro_batch_size=4,
        grad_accum_steps=8,
        max_lr=1e-3,
        warmup_steps=10,
        total_steps=100,
        decay_start_step=80,
        eval_interval_steps=50,
        eval_batches=5,
        checkpoint_interval_steps=50,
        log_interval_steps=5,
        seed=0,
        seq_len=2048,
    )
    assert cfg.effective_tokens_per_step(world_size=1) == 4 * 8 * 2048
    assert cfg.effective_tokens_per_step(world_size=4) == 4 * 8 * 2048 * 4
    assert cfg.effective_tokens_per_step() == 4 * 8 * 2048


def test_unknown_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        load_train_config(_write(tmp_path, REQUIRED_YAML + "bogus: 1\n"))


def test_missing_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="missing required keys"):
        load_train_config(_write(tmp_path, "seed: 1\n"))


def test_schedule_ordering_validated(tmp_path: Path) -> None:
    # decay_start_step > total_steps must be rejected.
    body = REQUIRED_YAML.replace("decay_start_step: 800", "decay_start_step: 1200")
    with pytest.raises(ConfigError, match="warmup_steps <= decay_start_step <= total_steps"):
        load_train_config(_write(tmp_path, body))


def test_warmup_after_decay_start_rejected(tmp_path: Path) -> None:
    body = REQUIRED_YAML.replace("warmup_steps: 100", "warmup_steps: 900")
    with pytest.raises(ConfigError, match="warmup_steps <= decay_start_step"):
        load_train_config(_write(tmp_path, body))


def test_min_lr_ratio_range_validated(tmp_path: Path) -> None:
    body = REQUIRED_YAML + "min_lr_ratio: 1.5\n"
    with pytest.raises(ConfigError, match="min_lr_ratio"):
        load_train_config(_write(tmp_path, body))


def test_non_positive_micro_batch_rejected(tmp_path: Path) -> None:
    body = REQUIRED_YAML.replace("micro_batch_size: 4", "micro_batch_size: 0")
    with pytest.raises(ConfigError, match="micro_batch_size"):
        load_train_config(_write(tmp_path, body))


def test_non_positive_grad_accum_rejected(tmp_path: Path) -> None:
    body = REQUIRED_YAML.replace("grad_accum_steps: 8", "grad_accum_steps: 0")
    with pytest.raises(ConfigError, match="grad_accum_steps"):
        load_train_config(_write(tmp_path, body))


def test_zero_grad_clip_rejected(tmp_path: Path) -> None:
    # 0 == "no clipping" in many frameworks, but here it would zero every
    # gradient and silently kill learning mid-run, so it must be rejected.
    with pytest.raises(ConfigError, match="grad_clip"):
        load_train_config(_write(tmp_path, REQUIRED_YAML + "grad_clip: 0\n"))


def test_beta1_out_of_range_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="beta1"):
        load_train_config(_write(tmp_path, REQUIRED_YAML + "beta1: 1.0\n"))


def test_beta2_out_of_range_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="beta2"):
        load_train_config(_write(tmp_path, REQUIRED_YAML + "beta2: 1.5\n"))


def test_negative_weight_decay_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="weight_decay"):
        load_train_config(_write(tmp_path, REQUIRED_YAML + "weight_decay: -0.1\n"))


def test_zero_checkpoint_interval_minutes_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="checkpoint_interval_minutes"):
        load_train_config(_write(tmp_path, REQUIRED_YAML + "checkpoint_interval_minutes: 0\n"))


def test_non_positive_eval_interval_rejected(tmp_path: Path) -> None:
    body = REQUIRED_YAML.replace("eval_interval_steps: 200", "eval_interval_steps: 0")
    with pytest.raises(ConfigError, match="eval_interval_steps"):
        load_train_config(_write(tmp_path, body))


def test_non_positive_eval_batches_rejected(tmp_path: Path) -> None:
    body = REQUIRED_YAML.replace("eval_batches: 20", "eval_batches: 0")
    with pytest.raises(ConfigError, match="eval_batches"):
        load_train_config(_write(tmp_path, body))


def test_non_positive_checkpoint_interval_steps_rejected(tmp_path: Path) -> None:
    body = REQUIRED_YAML.replace("checkpoint_interval_steps: 500", "checkpoint_interval_steps: 0")
    with pytest.raises(ConfigError, match="checkpoint_interval_steps"):
        load_train_config(_write(tmp_path, body))


def test_non_positive_log_interval_rejected(tmp_path: Path) -> None:
    body = REQUIRED_YAML.replace("log_interval_steps: 10", "log_interval_steps: 0")
    with pytest.raises(ConfigError, match="log_interval_steps"):
        load_train_config(_write(tmp_path, body))


def test_valid_tightened_optionals_load(tmp_path: Path) -> None:
    # A config that sits right at the edges of the new bounds still loads.
    body = REQUIRED_YAML + (
        "grad_clip: 0.5\nbeta1: 0.0\nbeta2: 0.999\nweight_decay: 0.0\n"
        "checkpoint_interval_minutes: 1\n"
    )
    cfg = load_train_config(_write(tmp_path, body))
    assert cfg.grad_clip == pytest.approx(0.5)
    assert cfg.beta1 == pytest.approx(0.0)
    assert cfg.weight_decay == pytest.approx(0.0)
    assert cfg.checkpoint_interval_minutes == 1


def test_shipped_base_400m_train_config_loads() -> None:
    cfg = load_train_config(TRAIN_CONFIG_DIR / "base_400m.yaml")
    assert cfg.model_config == Path("configs/model/base_400m.yaml")
    assert cfg.shard_index == Path("artifacts/shards-full/train/index.json")
    assert cfg.eval_shard_index == Path("artifacts/shards-full/eval/index.json")
    assert cfg.out_dir == Path("artifacts/checkpoints/base_400m")
    assert cfg.seq_len == 2048
    assert cfg.micro_batch_size == 4
    assert cfg.grad_accum_steps == 128
    assert cfg.max_lr == pytest.approx(3.0e-4)
    assert cfg.min_lr_ratio == pytest.approx(0.1)
    assert cfg.warmup_steps == 200
    # 20B two-epoch schedule: WSD decay starts at 80% of the step budget.
    assert cfg.decay_start_step == 15280
    assert cfg.total_steps == 19100
    assert cfg.eval_interval_steps == 250
    assert cfg.eval_batches == 50
    assert cfg.checkpoint_interval_steps == 40
    assert cfg.log_interval_steps == 5
    assert cfg.keep_checkpoints == 3
    assert cfg.seed == 1337
    assert cfg.wandb_project == "sqlpup-pretrain"
    assert cfg.wandb_run_name == "base400m-10b"
    # Load-bearing invariant: whatever mb/accum split the GPU probe picks, the
    # global batch must stay 1,048,576 tokens/step (0.5-1M spec band).
    assert cfg.effective_tokens_per_step() == 1_048_576


def test_shipped_base_400m_ddp_train_config_loads() -> None:
    # World-size-4 DDP twin of base_400m.yaml. grad_accum_steps is PER-RANK, so it
    # is halved 128 -> 32: 4 ranks x mb 4 x accum 32 x seq 2048 = 1,048,576, the
    # SAME global batch the single-GPU twin holds on one device with accum 128.
    cfg = load_train_config(TRAIN_CONFIG_DIR / "base_400m_ddp.yaml")
    assert cfg.grad_accum_steps == 32
    assert cfg.out_dir == Path("artifacts/checkpoints/base_400m_ddp")  # own ckpt namespace
    assert cfg.wandb_run_name == "base400m-10b-ddp4"
    # Load-bearing invariant: across the 4 ranks the global batch is 1,048,576
    # tokens/step, identical to the single-GPU twin (0.5-1M spec band).
    assert cfg.effective_tokens_per_step(world_size=4) == 1_048_576
    # Per rank (one process) it is a quarter of that.
    assert cfg.effective_tokens_per_step(world_size=1) == 262_144

    # It IS a twin: every field other than the three documented deltas must match
    # base_400m.yaml exactly, so the two configs cannot silently drift apart.
    twin = load_train_config(TRAIN_CONFIG_DIR / "base_400m.yaml")
    deltas = {"grad_accum_steps", "out_dir", "wandb_run_name"}
    for field in dataclasses.fields(TrainConfig):
        if field.name not in deltas:
            assert getattr(cfg, field.name) == getattr(twin, field.name), field.name


def test_draft_50m_config_pins_and_twin_invariant() -> None:
    config = load_train_config(Path("configs/train/base_50m_draft.yaml"))
    assert config.total_steps == 9222  # one epoch of the 9.67B corpus at 1M tok/step
    assert config.decay_start_step == 7378  # 80%
    assert config.micro_batch_size * config.grad_accum_steps * config.seq_len == 1_048_576
