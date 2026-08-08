"""GRPO configuration and the trainer-side guard.

Everything checkable without a GPU is checked here, so a misconfiguration fails
on a laptop in milliseconds instead of ten minutes into a rented box.

Two deliberate departures from TRL's defaults, both because of what this model
and this environment are:

**vLLM stays off.** vLLM 0.26 hard-pins ``torch==2.11`` against the 2.13 this
checkpoint was trained under, and rollouts for a 394M model with a 256-token
completion budget are cheap enough through ``transformers`` generate. Turning
it on would trade the run's largest integration risk for a speedup we do not
need at this size.

**A small KL anchor is on** (TRL 1.9 defaults ``beta`` to 0.0, the DAPO-style
setting). Removing the KL term helps when a policy must travel a long way, as
in long-chain reasoning. Here the risk runs the other direction: a 394M policy
on execution reward can degenerate faster than it improves, and the reference
model costs only ~0.8GB at this size. Cheap insurance, and it is the kind of
choice that must be disclosed rather than absorbed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from sqlpup.rl.health import RolloutHealth

# Small but non-zero: enough to anchor a 394M policy without preventing it from
# moving. Disclosed in the paper alongside the vLLM decision.
DEFAULT_BETA: float = 0.02
DEFAULT_CONTEXT_LIMIT: int = 2048


def build_grpo_config(
    *,
    out_dir: str | Path,
    num_generations: int = 4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_completion_length: int = 256,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    learning_rate: float = 1e-6,
    beta: float = DEFAULT_BETA,
    temperature: float = 1.0,
    num_processes: int = 1,
    **overrides: Any,
) -> Any:
    """A validated :class:`trl.GRPOConfig`.

    ``num_generations`` defaults to 4 rather than TRL's 8: the pre-registered
    pilot chose the smaller group to shrink the debugging surface and halve
    rollout cost. The headroom measurement that authorised this work was taken
    at k=8, so a full run may raise it -- that is a cost decision, not a
    methodological one.
    """
    from trl import GRPOConfig  # type: ignore[attr-defined]

    if num_generations < 2:
        raise ValueError(
            f"num_generations must be >= 2 for group-relative advantage, got {num_generations}"
        )
    effective_batch = num_processes * per_device_train_batch_size * gradient_accumulation_steps
    if effective_batch % num_generations:
        raise ValueError(
            f"effective batch {effective_batch} (processes {num_processes} x per-device "
            f"{per_device_train_batch_size} x accum {gradient_accumulation_steps}) must be "
            f"divisible by num_generations {num_generations}"
        )
    if max_completion_length >= context_limit:
        raise ValueError(
            f"max_completion_length {max_completion_length} leaves no room in a "
            f"context of {context_limit}"
        )

    return GRPOConfig(
        output_dir=str(out_dir),
        num_generations=num_generations,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_completion_length=max_completion_length,
        learning_rate=learning_rate,
        beta=beta,
        temperature=temperature,
        use_vllm=False,
        # Load-bearing: the reward reads gold_sql and db_path off each row.
        # TRL already defaults this to False for GRPO; set explicitly because a
        # silent True would zero every reward while training looked normal.
        remove_unused_columns=False,
        log_completions=True,
        **overrides,
    )


class RolloutGuard(TrainerCallback):
    """``TrainerCallback`` wrapper around :class:`RolloutHealth`.

    Subclassed rather than duck-typed: ``CallbackHandler`` dispatches *every*
    lifecycle event by attribute lookup, so an object defining only ``on_log``
    raises ``AttributeError`` on ``on_init_end`` before training starts. The
    decision logic stays in :class:`RolloutHealth`, which needs no trainer to
    test.
    """

    def __init__(self, health: RolloutHealth | None = None) -> None:
        self.health = health or RolloutHealth()
        self.halt_reason: str | None = None
        self.reward_errors = 0
        self.rollouts_seen = 0

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        reason = self.health.observe(
            logs or {},
            reward_errors=self.reward_errors,
            rollouts_seen=self.rollouts_seen,
        )
        if reason and self.halt_reason is None:
            self.halt_reason = reason
            control.should_training_stop = True
            print(f"HALT (pre-registered): {reason}", flush=True)
        return control
