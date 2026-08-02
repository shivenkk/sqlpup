"""Supervised fine-tuning data construction (torch-free).

Pairs are built on the SAME versioned prompt format evaluation uses
(:data:`sqlpup.eval.prompts.BIRD_DDL_V1`) -- the byte-identical train/eval
contract -- and ship as plain int tuples; the torch-backed trainer consumes
them on the training host.
"""

from __future__ import annotations

from sqlpup.sft.pairs import IGNORE_INDEX, BoundaryError, SFTPair, build_pair

__all__ = ["IGNORE_INDEX", "BoundaryError", "SFTPair", "build_pair"]
