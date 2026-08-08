"""Grammar-constrained decoding *interface* -- deliberately implementation-free.

Constrained SQL decoding will live inside the batched inference engine
(portfolio project #2), not inside any one generation loop here. This module
pins only the *shape* every future implementation and consumer agree on, so
generators can accept a constraint today without welding the feature to their
internals. Passing a constraint to a generator that cannot honor it must raise
``NotImplementedError`` -- never silently decode unconstrained.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class SQLConstraint(Protocol):
    """Token-level mask oracle: which vocabulary items may legally come next."""

    def allowed_token_mask(
        self, prefix_token_ids: Sequence[int], vocab_size: int
    ) -> Sequence[bool]:
        """Boolean mask of length ``vocab_size`` for the next-token choice."""
        ...
