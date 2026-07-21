"""Warmup-stable-decay (WSD) learning-rate schedule.

WSD has three phases: a linear warmup from 0 to ``max_lr``, a constant
``max_lr`` plateau, then a linear decay to ``min_lr = max_lr * min_lr_ratio``
over the tail. It is a *pure function* of ``(step, config)`` with no internal
state, which is exactly what makes it extendable mid-run: on resume you may
raise ``decay_start_step``/``total_steps`` and the plateau simply continues, no
schedule object to rewind. See :meth:`sqlpup.train.loop.Trainer.from_checkpoint`.

Warmup is 0-indexed: ``wsd_lr(0)`` is 0 and ``wsd_lr(warmup_steps)`` is
``max_lr`` (the plateau value). The optimizer applies this lr *before* taking
step ``step``.
"""

from __future__ import annotations

from sqlpup.train.config import TrainConfig


def wsd_lr(step: int, cfg: TrainConfig) -> float:
    """Return the learning rate for ``step`` under ``cfg``'s WSD schedule."""
    max_lr = cfg.max_lr
    min_lr = cfg.min_lr
    if step < cfg.warmup_steps:
        # Linear ramp 0 -> max_lr; the guard ensures warmup_steps > 0 here.
        return max_lr * step / cfg.warmup_steps
    if step < cfg.decay_start_step:
        return max_lr
    if step < cfg.total_steps:
        # Linear max_lr -> min_lr across [decay_start_step, total_steps]; the
        # guards guarantee decay_start_step < total_steps, so no zero divide.
        progress = (step - cfg.decay_start_step) / (cfg.total_steps - cfg.decay_start_step)
        return max_lr - (max_lr - min_lr) * progress
    # Past the schedule: clamp at the floor (safe if a run overshoots total).
    return min_lr
