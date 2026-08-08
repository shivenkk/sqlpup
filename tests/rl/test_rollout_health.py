"""Pre-registered halt conditions for the GRPO run.

Thresholds are fixed here *before* any rollout exists, for the same reason
every other gate in this project was: a threshold chosen after seeing the curve
is a description of the curve, not a test of it.

The failure this guards hardest is length collapse, because it is the one that
*looks like success*. Mean reward rises while the policy degenerates toward a
short, always-executable string. The reward's grounding requirement blocks the
most obvious such string, and this catches whatever else the policy invents.
"""

from __future__ import annotations

import pytest

from sqlpup.rl.health import (
    FLAT_GROUP_FRACTION,
    FLAT_GROUP_PATIENCE,
    LENGTH_COLLAPSE_RATIO,
    RolloutHealth,
)


def _healthy(step: int) -> dict[str, float]:
    return {
        "completions/mean_length": 80.0,
        "completions/clipped_ratio": 0.05,
        "reward": 0.3,
        "reward_std": 0.2,
        "frac_reward_zero_std": 0.4,
    }


def test_a_healthy_run_is_never_halted() -> None:
    health = RolloutHealth()
    for step in range(50):
        assert health.observe(_healthy(step)) is None


def test_length_collapse_halts_the_run() -> None:
    """The signature failure: completions shrink toward a trivial string."""
    health = RolloutHealth()
    for step in range(10):  # establish the baseline
        assert health.observe(_healthy(step)) is None

    collapsed = _healthy(0) | {"completions/mean_length": 80.0 * LENGTH_COLLAPSE_RATIO - 1}
    reasons = [health.observe(collapsed) for _ in range(5)]
    assert any(r and "length" in r.lower() for r in reasons), reasons


def test_a_brief_length_dip_is_tolerated() -> None:
    """One short batch is sampling noise, not collapse; halting on it would
    throw away a healthy run."""
    health = RolloutHealth()
    for _ in range(10):
        health.observe(_healthy(0))
    assert health.observe(_healthy(0) | {"completions/mean_length": 5.0}) is None
    assert health.observe(_healthy(0)) is None
    assert health.observe(_healthy(0)) is None


def test_every_group_flat_means_there_is_nothing_to_learn() -> None:
    """With group-relative advantage, a group whose rewards are all equal
    contributes exactly zero gradient. If that is nearly every group, the run
    is burning GPU hours to no effect and should stop rather than finish."""
    health = RolloutHealth()
    flat = _healthy(0) | {"frac_reward_zero_std": FLAT_GROUP_FRACTION + 0.01}
    reasons = [health.observe(flat) for _ in range(FLAT_GROUP_PATIENCE + 2)]
    assert any(r and "signal" in r.lower() for r in reasons), reasons


def test_flat_groups_early_on_are_tolerated_for_a_while() -> None:
    """Early training legitimately has many all-zero groups; the patience
    window exists so we do not kill a run before it warms up."""
    health = RolloutHealth()
    flat = _healthy(0) | {"frac_reward_zero_std": 1.0}
    assert health.observe(flat) is None


def test_mass_truncation_halts_the_run() -> None:
    """If most completions hit the cap they score 0 regardless of quality, so
    the gradient is noise and the completion budget is simply wrong."""
    health = RolloutHealth()
    for _ in range(10):
        health.observe(_healthy(0))
    reasons = [health.observe(_healthy(0) | {"completions/clipped_ratio": 0.9}) for _ in range(5)]
    assert any(r and "truncat" in r.lower() for r in reasons), reasons


def test_reward_errors_above_the_tolerance_halt_the_run() -> None:
    """A broken database or scorer would otherwise show up only as 'the model
    is not learning'."""
    health = RolloutHealth()
    for _ in range(10):
        health.observe(_healthy(0))
    reason = health.observe(_healthy(0), reward_errors=100, rollouts_seen=1000)
    assert reason and "error" in reason.lower()


def test_missing_metrics_are_ignored_rather_than_crashing() -> None:
    """TRL's log payload varies by version and by step (some keys only appear
    on evaluation steps). A KeyError here would kill the run it is meant to
    protect."""
    health = RolloutHealth()
    assert health.observe({"loss": 0.5}) is None
    assert health.observe({}) is None


@pytest.mark.parametrize("ratio", [0.0, 1.5])
def test_a_nonsensical_collapse_ratio_is_rejected(ratio: float) -> None:
    with pytest.raises(ValueError):
        RolloutHealth(length_collapse_ratio=ratio)


def test_reward_falling_while_the_policy_moves_is_flagged() -> None:
    """A specific divergence mode worth catching: the policy drifts (kl rising)
    while mean reward trends *down*, i.e. rollouts spread away from correctness
    rather than toward it. Sustained, that is the reward gradient losing to
    entropy, and the run is getting worse rather than noisier."""
    health = RolloutHealth()
    # Phase 1: the policy sits still and scores well (what pilot #2 looked like).
    for step in range(20):
        health.observe(_healthy(step) | {"reward": 0.55, "kl": 1.4e-4})
    # Phase 2: it starts moving, and reward falls as it moves.
    reason = None
    for step in range(21):
        reason = health.observe(
            _healthy(step) | {"reward": 0.55 - step * 0.02, "kl": 1.4e-4 * (1 + step)}
        )
    assert reason and "diverg" in reason.lower(), reason


def test_ordinary_reward_noise_is_not_flagged_as_divergence() -> None:
    """With a handful of prompts per step the mean swings hard; flagging that
    would halt every healthy run."""
    health = RolloutHealth()
    noisy = [0.5, 0.1, 0.6, 0.2, 0.55, 0.05, 0.5, 0.6, 0.15, 0.5] * 4
    out = [health.observe(_healthy(i) | {"reward": r, "kl": 1.4e-4}) for i, r in enumerate(noisy)]
    assert all(r is None or "diverg" not in r.lower() for r in out)


def test_kl_past_the_degeneration_ceiling_halts_the_run() -> None:
    """Uncharted territory for a 394M policy. The diagnostic peaked at 2.59e-2
    over 300 steps; 1,000 steps at the same settings could reach 0.05-0.08, and
    the literature puts degeneration near 0.1. The divergence guard only fires
    when reward *falls*, so a policy drifting far from its reference while
    reward merely stays flat would slip past it. This is the backstop."""
    from sqlpup.rl.health import KL_CEILING

    health = RolloutHealth()
    reasons = [health.observe(_healthy(i) | {"kl": KL_CEILING * 1.1}) for i in range(6)]
    assert any(r and "kl" in r.lower() for r in reasons), reasons


def test_high_but_sub_ceiling_kl_is_allowed() -> None:
    """The whole point of this run is to let the policy move. Halting at the
    diagnostic's own peak would forbid the movement we are paying for."""
    from sqlpup.rl.health import KL_CEILING

    health = RolloutHealth()
    for i in range(30):
        assert health.observe(_healthy(i) | {"kl": KL_CEILING * 0.4}) is None
