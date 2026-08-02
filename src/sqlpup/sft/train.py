"""Masked-pair SFT on the exported HF model (torch side of :mod:`sqlpup.sft`).

Everything data-shaped was decided upstream: pairs arrive tokenized with labels
already ``-100``-masked over the prompt (seam-verified at build time), so this
module is deliberately thin -- a JSONL dataset, a right-padding collator, and a
``transformers.Trainer`` invocation with the run's receipts returned to the
caller. Imported lazily by the CLI: this module needs the ``train`` extra.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class PairsDataset(Dataset[dict[str, Any]]):
    """The JSONL pair file, loaded eagerly as compact int32 tensors.

    Python int-lists cost ~36 bytes per token; at 789K pairs x ~1.1K tokens
    that is ~60GB, past what a 64GB training box can hold. int32 tensors cost
    4 bytes per token (~7GB for the same file); the collator's copy into the
    padded long batch casts for free.
    """

    def __init__(self, path: Path | str) -> None:
        self._rows: list[dict[str, torch.Tensor]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self._rows.append(
                    {
                        "input_ids": torch.tensor(row["input_ids"], dtype=torch.int32),
                        "labels": torch.tensor(row["labels"], dtype=torch.int32),
                    }
                )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]


def collate_pairs(batch: Sequence[dict[str, Any]], *, pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad to the batch max: pad ids, ``-100`` labels, 0 attention."""
    width = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for i, row in enumerate(batch):
        n = len(row["input_ids"])
        # direct assignment copies-and-casts from int32 rows (or lists) for free
        input_ids[i, :n] = row["input_ids"]
        labels[i, :n] = row["labels"]
        attention[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def _wandb_available() -> bool:
    if not os.environ.get("WANDB_API_KEY"):
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("WANDB_API_KEY set but wandb not installed (track extra); logging locally only")
        return False
    return True


def run_sft(
    *,
    pairs_path: Path | str,
    out_dir: Path | str,
    pad_id: int,
    model: Any | None = None,
    model_dir: Path | str | None = None,
    epochs: float = 1.0,
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 4,
    grad_accum: int = 16,
    warmup_ratio: float = 0.03,
    logging_steps: int = 20,
    seed: int = 0,
    save_final: bool = True,
    bf16: bool | None = None,
    save_steps: int | None = None,
    resume: bool = False,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Fine-tune ``model`` (or the model at ``model_dir``) on a pair file.

    Returns the run's receipts (examples, steps, final loss) for the manifest.
    Exactly one of ``model`` / ``model_dir`` must be provided -- the instance
    path exists so tests can train a tiny random model in seconds.
    """
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    if (model is None) == (model_dir is None):
        raise ValueError("provide exactly one of model or model_dir")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype="auto" if bf16 is None else torch.bfloat16
        )

    dataset = PairsDataset(pairs_path)
    use_bf16 = bool(bf16) if bf16 is not None else torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        # Spot boxes get reclaimed: periodic checkpoints make a 10h run resumable.
        save_strategy="no" if save_steps is None else "steps",
        save_steps=save_steps or 500,
        save_total_limit=1,
        bf16=use_bf16,
        seed=seed,
        # W&B when a key is present AND the package is installed (track extra);
        # a missing optional must degrade to local logging, never crash a run.
        report_to=["wandb"] if _wandb_available() else [],
        run_name=run_name,
        dataloader_drop_last=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_pairs(batch, pad_id=pad_id),
    )
    train_result = trainer.train(resume_from_checkpoint=resume or None)
    if save_final:
        trainer.save_model(str(out_dir))
        if model_dir is not None:
            # The output dir must be a complete, loadable artifact: carry the
            # tokenizer files over from the source model (Trainer only saves
            # them when it owns a processing_class, which this setup does not).
            import shutil

            for name in ("tokenizer.json", "tokenizer_config.json"):
                src = Path(model_dir) / name
                if src.exists():
                    shutil.copyfile(src, Path(out_dir) / name)

    return {
        "train_examples": len(dataset),
        "steps": int(train_result.global_step),
        "final_loss": float(train_result.training_loss),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "effective_batch": per_device_batch_size * grad_accum,
    }
