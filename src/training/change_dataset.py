from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChangePairDataset(Dataset):
    """
    Dataset for paired bi-temporal optical patches.

    Each item contains:
        before: [4, H, W]
        after:  [4, H, W]
        mask:   [H, W]
    """

    def __init__(
        self,
        root: str,
        indices: list[int] | None = None,
    ) -> None:
        self.root = Path(root)

        manifest_path = self.root / "manifest.npy"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest does not exist: {manifest_path}"
            )

        manifest = np.load(
            manifest_path,
            allow_pickle=True,
        )

        self.records = list(manifest)

        if indices is not None:
            self.records = [
                self.records[index]
                for index in indices
            ]

        if not self.records:
            raise ValueError(
                "Change dataset contains no records."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        record = self.records[index]

        before = np.load(record["t1"]).astype(
            np.float32
        )

        after = np.load(record["t2"]).astype(
            np.float32
        )

        mask = np.load(record["mask"]).astype(
            np.float32
        )

        if before.shape != after.shape:
            raise ValueError(
                "Before/after shapes do not match: "
                f"{before.shape} vs {after.shape}"
            )

        if before.ndim != 3:
            raise ValueError(
                f"Expected [C,H,W], got {before.shape}"
            )

        if before.shape[0] != 4:
            raise ValueError(
                f"Expected 4 channels, got {before.shape[0]}"
            )

        if mask.shape != before.shape[1:]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match "
                f"image shape {before.shape[1:]}"
            )

        if not np.isfinite(before).all():
            raise ValueError(
                "Before image contains non-finite values."
            )

        if not np.isfinite(after).all():
            raise ValueError(
                "After image contains non-finite values."
            )

        if not np.isfinite(mask).all():
            raise ValueError(
                "Mask contains non-finite values."
            )

        unique_mask = np.unique(mask)

        if not np.isin(
            unique_mask,
            [0.0, 1.0],
        ).all():
            raise ValueError(
                f"Mask must contain only 0/1, "
                f"found {unique_mask}"
            )

        return {
            "before": torch.from_numpy(before),
            "after": torch.from_numpy(after),
            "mask": torch.from_numpy(mask),
        }
