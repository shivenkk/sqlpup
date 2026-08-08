"""Run GRPO over the execution reward, and return a receipt.

Wiring only: the reward, the rollout rows, the pre-registered guard and the
config each live in their own module and are tested there. What this adds is
lifecycle -- one long-lived execution sandbox for the whole run, the guard
attached to the trainer, and a receipt recording what actually happened rather
than what was configured.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlpup.eval.execution import ExecutionScorer
from sqlpup.rl.reward import make_exact_match_reward, make_execution_reward
from sqlpup.rl.rollout_data import to_hf_dataset
from sqlpup.rl.setup import RolloutGuard, build_grpo_config


def run_grpo(
    *,
    model_dir: Path | str,
    rows: Sequence[dict[str, str]],
    out_dir: Path | str,
    num_generations: int = 4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_completion_length: int = 256,
    context_limit: int = 2048,
    learning_rate: float = 1e-6,
    beta: float | None = None,
    temperature: float = 1.0,
    max_steps: int = -1,
    timeout: float = 30.0,
    device: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Train and return ``{steps, rollouts_scored, reward_errors, halt_reason, ...}``."""
    if not rows:
        raise ValueError("no rollout rows: refusing to start a run with an empty dataset")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer  # type: ignore[attr-defined]

    from sqlpup.rl.setup import DEFAULT_BETA

    config_kwargs: dict[str, Any] = dict(
        out_dir=out_dir,
        num_generations=num_generations,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_completion_length=max_completion_length,
        context_limit=context_limit,
        learning_rate=learning_rate,
        beta=DEFAULT_BETA if beta is None else beta,
        reward_weights=[1.0, 0.0],
        temperature=temperature,
        **overrides,
    )
    if max_steps > 0:
        config_kwargs["max_steps"] = max_steps
    config = build_grpo_config(**config_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # Annotated Any: transformers wraps .to() in a decorator whose stub
    # declares the first argument as PreTrainedModel, so a device string or
    # torch.device both fail type-checking against it.
    model: Any = AutoModelForCausalLM.from_pretrained(str(model_dir))
    if device:
        model = model.to(torch.device(device))

    guard = RolloutGuard()
    # One sandbox for the whole run: the worker spawn cost is paid once rather
    # than per rollout, and every reward call shares its timeout semantics.
    with ExecutionScorer(timeout=timeout) as scorer:
        reward = make_execution_reward(scorer)
        # Second function, weight 0: TRL logs rollout EX under its own name each
        # step so bonus-farming is distinguishable from genuine sharpening.
        observe = make_exact_match_reward(scorer)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward, observe],
            args=config,
            train_dataset=to_hf_dataset(rows),
            processing_class=tokenizer,
            callbacks=[guard],
        )
        result = trainer.train()
        trainer.save_model(str(out_dir))

    return {
        "steps": int(getattr(result, "global_step", 0) or trainer.state.global_step),
        "rollouts_scored": len(rows) * num_generations,
        "reward_errors": int(getattr(reward, "errors", 0)),
        "halt_reason": guard.halt_reason,
        "baseline_completion_length": guard.health.baseline_length,
        "rows": len(rows),
        "num_generations": num_generations,
        "beta": config.beta,
        "learning_rate": config.learning_rate,
        "max_completion_length": config.max_completion_length,
    }


def write_receipt(path: Path | str, receipt: dict[str, Any]) -> None:
    """Write the run receipt to its own file.

    The first pilot redirected the command's stdout into ``receipt.json`` and
    produced a 20MB artefact: TRL logs per-step metrics and a rich completions
    table to stdout, so the run's own summary ended up buried at the end of
    them and the file would not parse as JSON. The receipt is the thing a
    later reader trusts, so it gets its own path.
    """
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
