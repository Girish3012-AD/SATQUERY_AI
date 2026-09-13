from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()

        if smooth <= 0:
            raise ValueError(
                "smooth must be positive."
            )

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(
                "Expected logits with shape [B,1,H,W], "
                f"got {tuple(logits.shape)}"
            )

        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        if targets.ndim != 4:
            raise ValueError(
                "Expected targets with shape [B,H,W] "
                "or [B,1,H,W], "
                f"got {tuple(targets.shape)}"
            )

        if logits.shape != targets.shape:
            raise ValueError(
                "Logits and targets must have identical shapes. "
                f"Got {tuple(logits.shape)} and "
                f"{tuple(targets.shape)}"
            )

        targets = targets.float()

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(
            probabilities.shape[0],
            -1,
        )

        targets = targets.reshape(
            targets.shape[0],
            -1,
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class BuildingSegmentationLoss(nn.Module):
    """
    Combined BCE + Dice loss.

    total_loss =
        bce_weight * BCEWithLogitsLoss
        +
        dice_weight * DiceLoss
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()

        if bce_weight < 0:
            raise ValueError(
                "bce_weight must be non-negative."
            )

        if dice_weight < 0:
            raise ValueError(
                "dice_weight must be non-negative."
            )

        if bce_weight + dice_weight <= 0:
            raise ValueError(
                "At least one loss weight must be positive."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if targets.ndim == 3:
            targets_for_bce = targets.unsqueeze(1)
        else:
            targets_for_bce = targets

        targets_for_bce = targets_for_bce.float()

        bce_loss = self.bce(
            logits,
            targets_for_bce,
        )

        dice_loss = self.dice(
            logits,
            targets_for_bce,
        )

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )
