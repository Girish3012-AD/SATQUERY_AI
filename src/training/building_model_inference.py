from pathlib import Path
from typing import Any

import torch

from src.training.building_model import BuildingUNet


def load_building_model(
    checkpoint_path: str,
    device: torch.device | None = None,
) -> BuildingUNet:
    """Load a trained BuildingUNet checkpoint."""

    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Building model checkpoint does not exist: {path}"
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    checkpoint: dict[str, Any] = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model = BuildingUNet(
        in_channels=int(checkpoint.get("in_channels", 4)),
        base_channels=int(checkpoint.get("base_channels", 16)),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def predict_building_mask(
    model: BuildingUNet,
    image: torch.Tensor,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Predict a binary building mask from one normalized image."""

    if image.ndim == 3:
        image_batch = image.unsqueeze(0)
    elif image.ndim == 4:
        image_batch = image
    else:
        raise ValueError(
            "Image must have shape [C,H,W] or [B,C,H,W]."
        )

    if image_batch.shape[1] != model.in_channels:
        raise ValueError(
            f"Expected {model.in_channels} channels, "
            f"got {image_batch.shape[1]}."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    if device is None:
        device = next(model.parameters()).device

    image_batch = image_batch.to(
        device=device,
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        logits = model(image_batch)
        probabilities = torch.sigmoid(logits)

    mask = (
        probabilities >= threshold
    ).to(torch.uint8)

    if mask.shape[0] == 1:
        return mask[0, 0].cpu()

    return mask[:, 0].cpu()
