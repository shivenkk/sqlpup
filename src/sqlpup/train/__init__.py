"""Pretraining loop for sqlpup.

Only the pure-Python surface is re-exported here: :class:`TrainConfig`,
:func:`load_train_config`, and the :func:`wsd_lr` schedule. The heavy
submodules (:mod:`sqlpup.train.data`, :mod:`sqlpup.train.checkpoint`,
:mod:`sqlpup.train.loop`) pull in ``torch`` (the optional ``train`` extra) and
are imported on demand, never at ``sqlpup`` package-import time.
"""

from __future__ import annotations

from sqlpup.train.config import TrainConfig, load_train_config
from sqlpup.train.lr import wsd_lr

__all__ = ["TrainConfig", "load_train_config", "wsd_lr"]
