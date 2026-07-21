"""Model definition for sqlpup.

Only :mod:`sqlpup.model.config` is imported here: it is pure-Python and depends
on no optional extras. The transformer itself lives in
:mod:`sqlpup.model.transformer` and pulls in ``torch`` (an optional ``train``
extra), so it is imported on demand rather than at package-import time.
"""

from __future__ import annotations

from sqlpup.model.config import ModelConfig, load_model_config

__all__ = ["ModelConfig", "load_model_config"]
