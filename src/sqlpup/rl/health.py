"""Pre-registered halt conditions for a GRPO run.

Every threshold here was fixed before the first rollout existed. A threshold
chosen after seeing the curve describes the curve; it does not test it.

The decision logic is deliberately a plain object rather than a
``TrainerCallback`` so it can be tested without a trainer. :class:`RolloutGuard`
is the thin adapter that lets TRL stop the run.

**Length collapse is the failure worth the most vigilance**, because it looks
like success from the metrics: mean reward climbs while the policy degenerates
toward a short string that always executes. The reward's grounding requirement
blocks the most obvious such string (``SELECT 1``); this catches whatever else
the policy finds.
"""

from __future__ import annotations

from typing import Any, Final

# Collapse = mean completion length falls under this fraction of the run's own
# early baseline. Generous, because SQL length varies a lot by question; the
# failure mode we are catching is a *collapse*, not a drift.
LENGTH_COLLAPSE_RATIO: Final = 0.35
LENGTH_COLLAPSE_PATIENCE: Final = 3

# A group whose rollouts all earn the same reward contributes exactly zero
# gradient under group-relative advantage. Nearly-all-flat means the run is
# burning GPU time for nothing.
FLAT_GROUP_FRACTION: Final = 0.95
FLAT_GROUP_PATIENCE: Final = 10

# Completions hitting the cap score 0 whatever the model knew, so the gradient
# is noise and the budget is simply misconfigured.
TRUNCATION_FRACTION: Final = 0.60
TRUNCATION_PATIENCE: Final = 3

# Reward-side exceptions are tolerated individually (one bad row must not end a
# long run) but not in bulk, which would mean the scorer or data is broken.
REWARD_ERROR_FRACTION: Final = 0.02

# Steps used to establish the length baseline before collapse can be declared.
BASELINE_STEPS: Final = 5

# Divergence: the policy is moving (kl rising off its own early level) while
# mean reward falls. That is the reward gradient losing to entropy -- rollouts
# spreading away from correctness rather than toward it. Compared over long
# windows because per-step reward is dominated by which prompts were drawn.
DIVERGENCE_WINDOW: Final = 20
DIVERGENCE_REWARD_DROP: Final = 0.15
DIVERGENCE_KL_RISE: Final = 3.0

# Absolute ceiling on how far the policy may drift from its reference. The
# divergence guard only fires when reward *falls*; a policy drifting a long way
# while reward merely stays flat would slip past it. The diagnostic peaked at
# 2.59e-2 over 300 steps, and the small-model RL literature puts degeneration
# near 0.1, so this is a backstop for territory we have not measured -- not a
# limit on the movement the run exists to produce.
KL_CEILING: Final = 0.1
KL_CEILING_PATIENCE: Final = 5


class RolloutHealth:
    """Consumes TRL's log payloads; returns a halt reason, or ``None``."""

    def __init__(
        self,
        *,
        length_collapse_ratio: float = LENGTH_COLLAPSE_RATIO,
        flat_group_fraction: float = FLAT_GROUP_FRACTION,
        truncation_fraction: float = TRUNCATION_FRACTION,
        reward_error_fraction: float = REWARD_ERROR_FRACTION,
    ) -> None:
        if not 0.0 < length_collapse_ratio < 1.0:
            raise ValueError(
                f"length_collapse_ratio must be in (0, 1), got {length_collapse_ratio}"
            )
        self._collapse_ratio = length_collapse_ratio
        self._flat_fraction = flat_group_fraction
        self._truncation_fraction = truncation_fraction
        self._error_fraction = reward_error_fraction

        self._rewards: list[float] = []
        self._kls: list[float] = []
        self._lengths: list[float] = []
        self._baseline: float | None = None
        self._short_streak = 0
        self._flat_streak = 0
        self._truncated_streak = 0
        self._high_kl_streak = 0

    @property
    def baseline_length(self) -> float | None:
        """Mean completion length over the run's first logged steps."""
        return self._baseline

    def observe(
        self,
        metrics: dict[str, Any],
        *,
        reward_errors: int = 0,
        rollouts_seen: int = 0,
    ) -> str | None:
        """Fold one log payload in; return a halt reason when one trips.

        Missing keys are ignored: TRL's payload varies by version and by step,
        and a ``KeyError`` here would kill the run this exists to protect.
        """
        if rollouts_seen and reward_errors / rollouts_seen > self._error_fraction:
            return (
                f"reward errors {reward_errors}/{rollouts_seen} exceed "
                f"{self._error_fraction:.0%} - the scorer or the data is broken"
            )

        length = _number(metrics.get("completions/mean_length"))
        if length is not None:
            if self._baseline is None:
                self._lengths.append(length)
                if len(self._lengths) >= BASELINE_STEPS:
                    self._baseline = sum(self._lengths) / len(self._lengths)
            else:
                floor = self._baseline * self._collapse_ratio
                self._short_streak = self._short_streak + 1 if length < floor else 0
                if self._short_streak >= LENGTH_COLLAPSE_PATIENCE:
                    return (
                        f"length collapse: mean completion {length:.0f} tokens is under "
                        f"{self._collapse_ratio:.0%} of the {self._baseline:.0f}-token baseline "
                        f"for {self._short_streak} consecutive logs"
                    )

        flat = _number(metrics.get("frac_reward_zero_std"))
        if flat is not None:
            self._flat_streak = self._flat_streak + 1 if flat > self._flat_fraction else 0
            if self._flat_streak > FLAT_GROUP_PATIENCE:
                return (
                    f"no learning signal: {flat:.0%} of groups have zero reward variance "
                    f"for {self._flat_streak} consecutive logs"
                )

        reward = _number(metrics.get("reward"))
        kl = _number(metrics.get("kl"))
        if kl is not None:
            self._high_kl_streak = self._high_kl_streak + 1 if kl > KL_CEILING else 0
            if self._high_kl_streak >= KL_CEILING_PATIENCE:
                return (
                    f"kl ceiling: {kl:.3f} exceeds {KL_CEILING} for "
                    f"{self._high_kl_streak} consecutive logs - the policy has drifted "
                    "further from its reference than this model size has been measured at"
                )
        if reward is not None:
            self._rewards.append(reward)
        if kl is not None:
            self._kls.append(kl)
        if len(self._rewards) >= 2 * DIVERGENCE_WINDOW and len(self._kls) >= 2 * DIVERGENCE_WINDOW:
            early_r = sum(self._rewards[:DIVERGENCE_WINDOW]) / DIVERGENCE_WINDOW
            late_r = sum(self._rewards[-DIVERGENCE_WINDOW:]) / DIVERGENCE_WINDOW
            early_k = sum(self._kls[:DIVERGENCE_WINDOW]) / DIVERGENCE_WINDOW
            late_k = sum(self._kls[-DIVERGENCE_WINDOW:]) / DIVERGENCE_WINDOW
            moved = early_k > 0 and late_k / early_k >= DIVERGENCE_KL_RISE
            if moved and (early_r - late_r) >= DIVERGENCE_REWARD_DROP:
                return (
                    f"divergence: mean reward fell {early_r:.2f} -> {late_r:.2f} while kl rose "
                    f"{early_k:.1e} -> {late_k:.1e} - the policy is moving away from correctness"
                )

        clipped = _number(metrics.get("completions/clipped_ratio"))
        if clipped is not None:
            self._truncated_streak = (
                self._truncated_streak + 1 if clipped > self._truncation_fraction else 0
            )
            if self._truncated_streak >= TRUNCATION_PATIENCE:
                return (
                    f"mass truncation: {clipped:.0%} of completions hit the length cap "
                    f"for {self._truncated_streak} consecutive logs"
                )
        return None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
