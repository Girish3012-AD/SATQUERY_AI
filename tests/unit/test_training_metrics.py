import pytest
import torch

from src.training.metrics import binary_segmentation_metrics


def test_perfect_prediction():
    targets = torch.tensor(
        [
            [
                [1, 0],
                [0, 1],
            ]
        ],
        dtype=torch.float32,
    )

    logits = torch.tensor(
        [
            [
                [
                    [10.0, -10.0],
                    [-10.0, 10.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    metrics = binary_segmentation_metrics(
        logits,
        targets,
    )

    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_complete_miss():
    targets = torch.ones(
        1,
        2,
        2,
    )

    logits = torch.full(
        (1, 1, 2, 2),
        -10.0,
    )

    metrics = binary_segmentation_metrics(
        logits,
        targets,
    )

    assert metrics["dice"] == pytest.approx(0.0)
    assert metrics["iou"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)


def test_partial_overlap():
    targets = torch.tensor(
        [
            [
                [1, 1],
                [0, 0],
            ]
        ],
        dtype=torch.float32,
    )

    logits = torch.tensor(
        [
            [
                [
                    [10.0, -10.0],
                    [10.0, -10.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    metrics = binary_segmentation_metrics(
        logits,
        targets,
    )

    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["iou"] == pytest.approx(1.0 / 3.0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_wrong_target_shape_rejected():
    logits = torch.randn(
        2,
        1,
        64,
        64,
    )

    targets = torch.randn(
        2,
        32,
        32,
    )

    with pytest.raises(ValueError):
        binary_segmentation_metrics(
            logits,
            targets,
        )


def test_invalid_threshold_rejected():
    logits = torch.randn(
        1,
        1,
        4,
        4,
    )

    targets = torch.zeros(
        1,
        4,
        4,
    )

    with pytest.raises(ValueError):
        binary_segmentation_metrics(
            logits,
            targets,
            threshold=1.0,
        )
