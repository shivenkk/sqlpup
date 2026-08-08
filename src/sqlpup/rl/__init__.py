"""Execution-feedback RL components (torch-free reward; the trainer is TRL's).

The reward is the same contract the eval harness scores with -- official BIRD
set-equality via the killable sandbox -- so optimizing it optimizes the
benchmark metric itself, not a proxy.
"""

from __future__ import annotations

from sqlpup.rl.reward import EXECUTES_BONUS, execution_reward

__all__ = ["EXECUTES_BONUS", "execution_reward"]
