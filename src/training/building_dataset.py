from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class SpaceNetBuildingDataset(Dataset):
    """
    PyTorch dataset for SpaceNet building segmentation patches.

    Image:
        [4, H, W], float32

    Mask:
        [H, W], float32 with values 0 or 1
    """

    def __init__(
        self,
        manifest_path: str,
        include_empty: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest does not exist: {self.manifest_path}"
            )

        records = np.load(
            self.manifest_path,
            allow_pickle=True,
        ).tolist()

        if not isinstance(records, list):
            raise ValueError("Manifest must contain a list of records.")

        if not records:
            raise ValueError("Manifest contains no records.")

        if include_empty:
            self.records = records
        else:
            self.records = [
                record
                for record in records
                if record["building_pixels"] > 0
            ]

        if not self.records:
            raise ValueError(
                "No usable records remain after filtering."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]

        image_path = Path(record["image"])
        mask_path = Path(record["mask"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image patch does not exist: {image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask patch does not exist: {mask_path}"
            )

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if image.ndim != 3:
            raise ValueError(
                f"Expected image shape [C,H,W], got {image.shape}"
            )

        if mask.ndim != 2:
            raise ValueError(
                f"Expected mask shape [H,W], got {mask.shape}"
            )

        if image.shape[1:] != mask.shape:
            raise ValueError(
                f"Image/mask spatial dimensions do not match: "
                f"{image.shape} vs {mask.shape}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"Image contains non-finite values: {image_path}"
            )

        unique_mask_values = set(np.unique(mask).tolist())

        if not unique_mask_values.issubset({0.0, 1.0}):
            raise ValueError(
                "Mask must contain only 0 and 1. "
                f"Found: {sorted(unique_mask_values)}"
            )

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor

    def record(self, index: int) -> dict[str, Any]:
        return self.records[index]
