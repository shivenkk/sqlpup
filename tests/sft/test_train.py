"""SFT trainer wiring: dataset, padding collator, and a tiny end-to-end run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sqlpup.sft.train import PairsDataset, collate_pairs, run_sft  # noqa: E402


def _write_pairs(path: Path, sizes: list[int]) -> None:
    with path.open("w") as f:
        for n in sizes:
            ids = list(range(2, 2 + n))
            labels = [-100] * (n // 2) + ids[n // 2 :]
            f.write(
                json.dumps({"input_ids": ids, "labels": labels, "prompt_tokens": n // 2}) + "\n"
            )


def test_dataset_loads_jsonl_as_compact_tensors(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_pairs(path, [8, 5])
    ds = PairsDataset(path)
    assert len(ds) == 2
    assert ds[0]["input_ids"].dtype == torch.int32  # 4 bytes/token, not a Python list
    assert ds[0]["input_ids"].tolist() == list(range(2, 10))
    assert ds[1]["labels"].tolist()[:2] == [-100, -100]


def test_collate_left_pads_nothing_and_right_pads_batch(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_pairs(path, [8, 5])
    ds = PairsDataset(path)
    batch = collate_pairs([ds[0], ds[1]], pad_id=0)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)
    assert batch["attention_mask"].tolist()[1] == [1] * 5 + [0] * 3
    # padding positions carry no loss and pad ids
    assert batch["input_ids"][1, 5:].tolist() == [0, 0, 0]
    assert batch["labels"][1, 5:].tolist() == [-100, -100, -100]
    # prompt masking preserved from the file
    assert batch["labels"][0, 0].item() == -100


def test_run_sft_trains_and_saves_a_tiny_model(tmp_path: Path) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    path = tmp_path / "pairs.jsonl"
    _write_pairs(path, [12, 9, 7, 11])
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)  # type: ignore[no-untyped-call]
    out_dir = tmp_path / "out"
    result = run_sft(
        model=model,
        pairs_path=path,
        out_dir=out_dir,
        pad_id=0,
        epochs=1,
        learning_rate=1e-3,
        per_device_batch_size=2,
        grad_accum=1,
        logging_steps=1,
        save_final=True,
    )
    assert (out_dir / "model.safetensors").exists() or (out_dir / "pytorch_model.bin").exists()
    assert result["train_examples"] == 4
    assert result["steps"] >= 2
