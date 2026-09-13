from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.training.building_dataset import SpaceNetBuildingDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "data/remote_sensing/spacenet4/patches/"
    "Atlanta_743501_3721539/manifest.npy"
)


@pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="SpaceNet patch dataset is not available",
)
def test_dataset_length():
    dataset = SpaceNetBuildingDataset(str(MANIFEST))

    assert len(dataset) == 196


@pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="SpaceNet patch dataset is not available",
)
def test_dataset_sample_shapes_and_types():
    dataset = SpaceNetBuildingDataset(str(MANIFEST))

    image, mask = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    assert image.shape == (4, 64, 64)
    assert mask.shape == (64, 64)

    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32


@pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="SpaceNet patch dataset is not available",
)
def test_dataset_values_are_valid():
    dataset = SpaceNetBuildingDataset(str(MANIFEST))

    for index in [0, 50, 100, 150]:
        image, mask = dataset[index]

        assert torch.isfinite(image).all()
        assert torch.isfinite(mask).all()

        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0

        unique_values = torch.unique(mask).tolist()

        assert set(unique_values).issubset({0.0, 1.0})


@pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="SpaceNet patch dataset is not available",
)
def test_dataset_can_filter_empty_patches():
    dataset = SpaceNetBuildingDataset(
        str(MANIFEST),
        include_empty=False,
    )

    assert len(dataset) == 88

    for index in [0, len(dataset) // 2, len(dataset) - 1]:
        _, mask = dataset[index]

        assert torch.count_nonzero(mask) > 0


@pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="SpaceNet patch dataset is not available",
)
def test_dataloader_batch():
    dataset = SpaceNetBuildingDataset(str(MANIFEST))

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    assert images.shape == (8, 4, 64, 64)
    assert masks.shape == (8, 64, 64)

    assert images.dtype == torch.float32
    assert masks.dtype == torch.float32


@pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="SpaceNet patch dataset is not available",
)
def test_manifest_record_access():
    dataset = SpaceNetBuildingDataset(str(MANIFEST))

    record = dataset.record(0)

    assert record["patch_id"] == 0
    assert "image" in record
    assert "mask" in record
    assert "building_pixels" in record
    assert "building_fraction" in record
