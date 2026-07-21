"""Atomic checkpointing with exact-resume state and pruning.

One checkpoint is a single ``torch.save`` dict (model + optimizer state, the
step, the dataloader ``(epoch, cursor)`` position, RNG states, and a config
snapshot). Writes are atomic: the payload lands in ``step_{N}.pt.tmp`` and is
renamed to ``step_{N}.pt`` only once fully written, so an interrupted save never
leaves a torn file a resume would choke on. A small ``latest.json`` pointer
tracks the newest checkpoint and the protected stage-final slot.

The checkpoint *directory* is the unit of external sync: at AWS time it is
mirrored to S3 wholesale (no boto3 here -- that is account-gated and lives in a
later task). Filenames in ``latest.json`` are stored relative to the directory
so the whole folder relocates cleanly.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

_LATEST = "latest.json"


def gather_rng_states() -> dict[str, Any]:
    """Snapshot the process-global RNGs for exact resume.

    The dataloader derives its shuffle from ``(seed, epoch)`` and resumes from
    the saved ``(epoch, cursor)`` position, so it needs no generator snapshot;
    these global states cover any *other* stochastic op in a step (e.g. dropout,
    if added later). CUDA states are included when present; MPS RNG is not
    captured (resume is exercised on CPU/CUDA -- revisit if MPS resume is needed).
    """
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict[str, Any]) -> None:
    """Restore RNGs captured by :func:`gather_rng_states`.

    CUDA states are restored per device, sliced to ``min(saved, present)``, so
    a checkpoint moves across world sizes (e.g. a 4-GPU DDP snapshot resuming
    on a single-GPU box): extra saved states are ignored and extra devices
    keep their fresh generators. ``set_rng_state_all`` would raise IndexError
    whenever the checkpoint carries more states than the machine has devices.
    Bit-exact RNG continuity is only meaningful at the same world size anyway;
    a cross-world resume is deliberate and the model/optimizer state is what
    must carry over exactly.
    """
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available():
        saved = list(states.get("cuda", []))
        for device_index in range(min(len(saved), torch.cuda.device_count())):
            torch.cuda.set_rng_state(saved[device_index], device_index)


def save_checkpoint(
    out_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_config_snapshot: dict[str, Any],
    data_position: tuple[int, int],
    rng_states: dict[str, Any],
) -> Path:
    """Atomically write a checkpoint and update ``latest.json``; return its path.

    ``model`` must be the unwrapped module (the caller unwraps DDP), so the saved
    keys match a fresh model's ``state_dict`` on resume.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_config": train_config_snapshot,
        "data_position": tuple(data_position),
        "rng": rng_states,
    }
    final = out_dir / f"step_{step}.pt"
    tmp = out_dir / f"step_{step}.pt.tmp"
    _sweep_stale_tmp(out_dir, tmp)
    torch.save(payload, tmp)
    tmp.replace(final)  # atomic rename within the directory
    _update_latest(out_dir, step, final.name)
    return final


def _sweep_stale_tmp(out_dir: Path, keep: Path) -> None:
    """Delete orphaned ``step_*.pt.tmp`` files from crashed/spot-killed saves.

    A save that dies between :func:`torch.save` and the atomic rename leaves a
    ``step_{N}.pt.tmp`` behind; across a 30h spot run these orphans accumulate
    to GB scale. Each successful save sweeps them before writing its own tmp.
    Only ``step_*.pt.tmp`` is matched -- real ``step_*.pt`` checkpoints and
    ``latest.json`` are never touched -- and ``keep`` (the tmp this save is
    about to write) is preserved. The matches are materialised before unlinking
    so the directory is not mutated mid-scan.
    """
    for stale in list(out_dir.glob("step_*.pt.tmp")):
        if stale != keep:
            stale.unlink()


def load_checkpoint(path_or_dir: Path) -> dict[str, Any]:
    """Load a checkpoint dict. A directory resolves to its ``latest.json`` target.

    ``weights_only=False`` because the payload holds trusted Python objects (RNG
    states, the config snapshot) beyond plain tensors; these checkpoints are
    self-produced, never untrusted input.
    """
    path = Path(path_or_dir)
    if path.is_dir():
        meta = json.loads((path / _LATEST).read_text(encoding="utf-8"))
        path = path / str(meta["path"])
    loaded: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    return loaded


def mark_stage_final(out_dir: Path, step: int) -> None:
    """Mark ``step_{step}.pt`` as the stage-final checkpoint (pruning-protected)."""
    meta = _read_latest(out_dir)
    meta["stage_final"] = f"step_{step}.pt"
    _write_latest(out_dir, meta)


def prune_checkpoints(out_dir: Path, keep_last: int = 3) -> None:
    """Delete all but the newest ``keep_last`` checkpoints (by step number).

    The file named in ``latest.json``'s ``stage_final`` slot is never deleted,
    even if it falls outside the newest ``keep_last``.
    """
    files = sorted(out_dir.glob("step_*.pt"), key=_step_number)
    keep = set(files[-keep_last:]) if keep_last > 0 else set()
    final = _read_latest(out_dir).get("stage_final")
    if final:
        keep.add(out_dir / str(final))
    for path in files:
        if path not in keep:
            path.unlink()


def _step_number(path: Path) -> int:
    # "step_10.pt" -> 10; numeric sort so step_10 orders after step_9.
    return int(path.stem.split("_")[-1])


def _read_latest(out_dir: Path) -> dict[str, Any]:
    latest = out_dir / _LATEST
    if not latest.exists():
        return {}
    parsed: dict[str, Any] = json.loads(latest.read_text(encoding="utf-8"))
    return parsed


def _update_latest(out_dir: Path, step: int, filename: str) -> None:
    meta = _read_latest(out_dir)
    meta["step"] = step
    meta["path"] = filename
    meta["wall_time"] = time.time()
    meta.setdefault("stage_final", None)  # preserve an existing value
    _write_latest(out_dir, meta)


def _write_latest(out_dir: Path, meta: dict[str, Any]) -> None:
    latest = out_dir / _LATEST
    tmp = latest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(latest)  # atomic, so a reader never sees a half-written pointer
