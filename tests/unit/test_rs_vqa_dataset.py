import json
from pathlib import Path
import pytest

from src.training.build_rs_vqa_dataset import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    build_dataset,
)


def test_build_rs_vqa_dataset(tmp_path):
    if not DEFAULT_MANIFEST.exists():
        pytest.skip(
            f"DATASET_REQUIRED: SpaceNet4 patch manifest not found at {DEFAULT_MANIFEST}"
        )

    output = tmp_path / "rs_vqa"

    metadata = build_dataset(
        output_dir=output,
    )

    assert metadata["total_patches"] == 196
    assert metadata["training_patches"] > 0
    assert metadata["validation_patches"] > 0
    assert metadata["train_validation_patch_overlap"] == 0

    train_path = output / "train.jsonl"
    val_path = output / "val.jsonl"

    assert train_path.exists()
    assert val_path.exists()

    train = [
        json.loads(line)
        for line in train_path.read_text().splitlines()
    ]

    val = [
        json.loads(line)
        for line in val_path.read_text().splitlines()
    ]

    assert len(train) == metadata["training_examples"]
    assert len(val) == metadata["validation_examples"]

    assert len(train) == metadata["training_patches"] * 4
    assert len(val) == metadata["validation_patches"] * 4

    train_images = {
        example["image"]
        for example in train
    }

    val_images = {
        example["image"]
        for example in val
    }

    assert train_images.isdisjoint(val_images)

    for example in train + val:
        assert example["source"] == "SpaceNet4"
        assert example["task"] in {
            "building_presence",
            "building_density",
            "building_coverage",
            "scene_type",
        }
        assert example["question"]
        assert example["answer"]
        assert example["ground_truth"]
        assert example["ground_truth"]["building_pixels"] >= 0
        assert (
            0.0
            <= example["ground_truth"]["building_fraction"]
            <= 1.0
        )

    image_dir = output / "images"

    assert len(list(image_dir.glob("*.png"))) == 196


def test_default_output_is_defined():
    assert isinstance(DEFAULT_OUTPUT, Path)
