from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.training.building_model import BuildingUNet


def load_building_model(
    checkpoint_path: str,
    device: str | None = None,
) -> tuple[BuildingUNet, dict[str, Any]]:
    """
    Load a trained BuildingUNet checkpoint.

    Returns:
        model: loaded model in evaluation mode
        metadata: checkpoint metadata
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model = BuildingUNet(
        in_channels=checkpoint["in_channels"],
        base_channels=checkpoint["base_channels"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }

    metadata["device"] = device
    metadata["checkpoint_path"] = str(path.resolve())

    return model, metadata


@torch.no_grad()
def predict_building_mask(
    model: BuildingUNet,
    image: np.ndarray,
    device: str | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict a binary building mask.

    Args:
        model:
            Loaded BuildingUNet.
        image:
            Float32 image with shape [C,H,W].
            Expected values approximately in [0,1].
        device:
            Target device.
        threshold:
            Probability threshold for building prediction.

    Returns:
        probability:
            Float32 array [H,W].
        binary_mask:
            uint8 array [H,W], values 0 or 1.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    if image.ndim != 3:
        raise ValueError(
            f"image must have shape [C,H,W], got {image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError("image contains non-finite values")

    expected_channels = model.in_channels

    if image.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels, "
            f"got {image.shape[0]}"
        )

    if device is None:
        device = next(model.parameters()).device.type

    tensor = torch.from_numpy(
        image.astype(np.float32)
    ).unsqueeze(0).to(device)

    logits = model(tensor)

    probability = torch.sigmoid(
        logits
    )[0, 0].cpu().numpy()

    binary_mask = (
        probability >= threshold
    ).astype(np.uint8)

    return probability, binary_mask
