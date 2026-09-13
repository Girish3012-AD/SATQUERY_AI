import pytest
import torch

from src.training.building_loss import BuildingSegmentationLoss
from src.training.building_model import BuildingUNet


@pytest.mark.parametrize(
    "batch_size",
    [1, 2],
)
def test_building_unet_output_shape(batch_size):
    model = BuildingUNet(
        in_channels=4,
        base_channels=16,
    )

    x = torch.randn(
        batch_size,
        4,
        64,
        64,
    )

    output = model(x)

    assert output.shape == (
        batch_size,
        1,
        64,
        64,
    )

    assert torch.isfinite(output).all()


def test_building_unet_rejects_wrong_channels():
    model = BuildingUNet(
        in_channels=4,
        base_channels=16,
    )

    x = torch.randn(
        2,
        3,
        64,
        64,
    )

    with pytest.raises(ValueError):
        model(x)


def test_building_unet_rejects_wrong_dimensions():
    model = BuildingUNet(
        in_channels=4,
        base_channels=16,
    )

    x = torch.randn(
        4,
        64,
        64,
    )

    with pytest.raises(ValueError):
        model(x)


def test_building_segmentation_loss_is_finite():
    loss_fn = BuildingSegmentationLoss()

    logits = torch.randn(
        2,
        1,
        64,
        64,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        2,
        (2, 64, 64),
    ).float()

    loss = loss_fn(
        logits,
        targets,
    )

    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_model_loss_backward():
    model = BuildingUNet(
        in_channels=4,
        base_channels=16,
    )

    loss_fn = BuildingSegmentationLoss()

    images = torch.randn(
        2,
        4,
        64,
        64,
    )

    targets = torch.randint(
        0,
        2,
        (2, 64, 64),
    ).float()

    logits = model(images)
    loss = loss_fn(
        logits,
        targets,
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )

    total_gradient_norm = sum(
        float(gradient.detach().norm())
        for gradient in gradients
    )

    assert total_gradient_norm > 0.0
