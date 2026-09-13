from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(
            probabilities.shape[0],
            -1,
        )

        target = target.reshape(
            target.shape[0],
            -1,
        )

        intersection = (
            probabilities * target
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + target.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return (1.0 - dice).mean()


class ChangeLoss(nn.Module):
    """
    Combined BCE + Dice loss.

    BCE stabilizes pixel classification.
    Dice helps with the foreground/background imbalance.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
    ) -> None:
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice = DiceLoss()

        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = target.float()

        dice_loss = self.dice(
            logits,
            target,
        )

        bce_loss = self.bce(
            logits,
            target,
        )

        return (
            self.dice_weight * dice_loss
            + self.bce_weight * bce_loss
        )
