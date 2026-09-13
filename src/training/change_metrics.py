from __future__ import annotations

import torch


def binary_change_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:

    if target.ndim == 3:
        target = target.unsqueeze(1)

    probabilities = torch.sigmoid(logits)

    prediction = (
        probabilities >= threshold
    ).float()

    target = target.float()

    prediction = prediction.reshape(-1)
    target = target.reshape(-1)

    tp = float(
        ((prediction == 1) & (target == 1)).sum()
    )

    fp = float(
        ((prediction == 1) & (target == 0)).sum()
    )

    fn = float(
        ((prediction == 0) & (target == 1)).sum()
    )

    tn = float(
        ((prediction == 0) & (target == 0)).sum()
    )

    dice = (
        2.0 * tp
        / max(2.0 * tp + fp + fn, 1.0)
    )

    iou = (
        tp
        / max(tp + fp + fn, 1.0)
    )

    precision = (
        tp
        / max(tp + fp, 1.0)
    )

    recall = (
        tp
        / max(tp + fn, 1.0)
    )

    accuracy = (
        (tp + tn)
        / max(tp + tn + fp + fn, 1.0)
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }
