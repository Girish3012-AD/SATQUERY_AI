from pathlib import Path

import torch

from src.training.rs_vqa_lora_collator import (
    DEFAULT_QWEN_SNAPSHOT,
    RSVQACollator,
)
from src.training.rs_vqa_lora_data import (
    DEFAULT_DATASET_ROOT,
    RSVQADataset,
)


def test_qwen_snapshot_exists():
    assert Path(DEFAULT_QWEN_SNAPSHOT).exists()
    assert (
        Path(DEFAULT_QWEN_SNAPSHOT) / "config.json"
    ).exists()


def test_processor_loads():
    collator = RSVQACollator()

    assert collator.processor is not None


def test_single_example_processing():
    dataset = RSVQADataset(
        DEFAULT_DATASET_ROOT / "train.jsonl"
    )

    collator = RSVQACollator()

    batch = collator([dataset[0]])

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    assert isinstance(
        batch["input_ids"],
        torch.Tensor,
    )

    assert isinstance(
        batch["labels"],
        torch.Tensor,
    )

    assert (
        batch["input_ids"].shape
        == batch["labels"].shape
    )


def test_prompt_tokens_are_masked():
    dataset = RSVQADataset(
        DEFAULT_DATASET_ROOT / "train.jsonl"
    )

    collator = RSVQACollator()

    batch = collator([dataset[0]])

    labels = batch["labels"]

    masked = (labels == -100).sum().item()
    learnable = (labels != -100).sum().item()

    assert masked > 0
    assert learnable > 0
