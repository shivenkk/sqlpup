#!/usr/bin/env python
"""Generate LR-grid sweep configs from a base train config.

Derives one train YAML per learning rate from a base config (e.g.
``configs/train/proxy_30m.yaml``), setting ``max_lr`` and a distinct ``out_dir``
(and, when the base enables tracking, a distinct ``wandb_run_name``) while
copying every other field verbatim. The GPU sweep runner then trains each
generated config; picking the actual grid values is the runner's job.

    python scripts/make_sweep_configs.py \\
        --config configs/train/proxy_30m.yaml \\
        --lrs 1.5e-3 3e-3 6e-3 \\
        --out-dir artifacts/sweep

Stdlib + pyyaml only (torch-free); the derivation core is import-testable, like
``scripts/smoke_train.py``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def variant_name(base_name: str, lr: str) -> str:
    """Run/variant identifier for one LR, e.g. ``proxy_30m-lr1.5e-3``.

    ``lr`` is the raw grid string (not a reparsed float) so the tag keeps the
    author's formatting (``1.5e-3`` stays ``1.5e-3``, not ``0.0015``).
    """
    return f"{base_name}-lr{lr}"


def variant_filename(base_name: str, lr: str) -> str:
    """Output YAML filename for one LR variant."""
    return f"{variant_name(base_name, lr)}.yaml"


def derive_variant(base: Mapping[str, Any], base_name: str, lr: str) -> dict[str, Any]:
    """Return a copy of ``base`` retargeted to learning rate ``lr``.

    Changes exactly three things: ``max_lr`` (to ``float(lr)``), ``out_dir`` (a
    per-LR suffix so runs never collide), and -- only when the base sets
    ``wandb_project`` -- ``wandb_run_name``. Every other field is copied verbatim
    and the input mapping is left unmutated.
    """
    variant = dict(base)
    variant["max_lr"] = float(lr)
    variant["out_dir"] = f"{base['out_dir']}-lr{lr}"
    if base.get("wandb_project") is not None:
        variant["wandb_run_name"] = variant_name(base_name, lr)
    return variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_sweep_configs", description="derive LR-grid train configs from a base config"
    )
    parser.add_argument("--config", required=True, help="base train config YAML")
    parser.add_argument(
        "--lrs",
        nargs="+",
        required=True,
        metavar="LR",
        help="learning rates for the grid (kept as written for run names, e.g. 1.5e-3 3e-3)",
    )
    parser.add_argument("--out-dir", required=True, help="directory to write the generated configs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = Path(args.config)
    base: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_name = config_path.stem

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for lr in args.lrs:
        variant = derive_variant(base, base_name, lr)
        path = out_dir / variant_filename(base_name, lr)
        # sort_keys=False preserves the base field order (matches smoke_train's
        # derived-config writer); YAML comments in the base are not carried over.
        path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        written.append(str(path))

    print(json.dumps({"config": str(config_path), "lrs": list(args.lrs), "written": written}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
