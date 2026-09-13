from __future__ import annotations

import torch


def binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calculate binary segmentation metrics.

    Returns:
        dice
        iou
        precision
        recall
    """

    if logits.ndim != 4:
        raise ValueError(
            "Expected logits shape [B,1,H,W], "
            f"got {tuple(logits.shape)}"
        )

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    if targets.ndim != 4:
        raise ValueError(
            "Expected targets shape [B,H,W] or [B,1,H,W], "
            f"got {tuple(targets.shape)}"
        )

    if logits.shape != targets.shape:
        raise ValueError(
            "Logits and targets must have identical shapes."
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    targets_bool = targets.bool()

    predictions = predictions.reshape(-1)
    targets_bool = targets_bool.reshape(-1)

    true_positive = torch.logical_and(
        predictions,
        targets_bool,
    ).sum().float()

    false_positive = torch.logical_and(
        predictions,
        ~targets_bool,
    ).sum().float()

    false_negative = torch.logical_and(
        ~predictions,
        targets_bool,
    ).sum().float()

    epsilon = 1e-8

    dice = (
        2.0 * true_positive
        / (
            2.0 * true_positive
            + false_positive
            + false_negative
            + epsilon
        )
    )

    iou = (
        true_positive
        / (
            true_positive
            + false_positive
            + false_negative
            + epsilon
        )
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
            + epsilon
        )
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
            + epsilon
        )
    )

    return {
        "dice": float(dice.detach()),
        "iou": float(iou.detach()),
        "precision": float(precision.detach()),
        "recall": float(recall.detach()),
    }
