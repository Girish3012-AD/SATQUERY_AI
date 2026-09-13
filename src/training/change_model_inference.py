from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.training.change_model import ChangeUNet


def load_change_model(
    checkpoint_path: str,
    device: str | torch.device | None = None,
) -> tuple[ChangeUNet, torch.device]:
    """
    Load a trained ChangeUNet checkpoint.

    Returns:
        model
        device
    """
    checkpoint = Path(checkpoint_path)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint}"
        )

    if device is None:
        target_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        target_device = torch.device(device)

    payload = torch.load(
        checkpoint,
        map_location=target_device,
        weights_only=False,
    )

    model = ChangeUNet(
        in_channels=int(
            payload.get("in_channels", 4)
        ),
        base_channels=int(
            payload.get("base_channels", 16)
        ),
    )

    model.load_state_dict(
        payload["state_dict"]
    )

    model.to(target_device)
    model.eval()

    return model, target_device


def predict_change_mask(
    model: ChangeUNet,
    before: np.ndarray | torch.Tensor,
    after: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict a change probability map and binary mask.

    Input:
        [4,H,W] or [B,4,H,W]

    Returns:
        probability: float32 numpy array
        mask: uint8 numpy array
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if isinstance(before, np.ndarray):
        before_tensor = torch.from_numpy(
            before.astype(np.float32)
        )
    else:
        before_tensor = before.float()

    if isinstance(after, np.ndarray):
        after_tensor = torch.from_numpy(
            after.astype(np.float32)
        )
    else:
        after_tensor = after.float()

    if before_tensor.ndim == 3:
        before_tensor = before_tensor.unsqueeze(0)

    if after_tensor.ndim == 3:
        after_tensor = after_tensor.unsqueeze(0)

    if before_tensor.ndim != 4:
        raise ValueError(
            f"Expected before [B,C,H,W], "
            f"got {tuple(before_tensor.shape)}"
        )

    if after_tensor.ndim != 4:
        raise ValueError(
            f"Expected after [B,C,H,W], "
            f"got {tuple(after_tensor.shape)}"
        )

    if before_tensor.shape != after_tensor.shape:
        raise ValueError(
            "Before/after tensor shapes differ: "
            f"{tuple(before_tensor.shape)} vs "
            f"{tuple(after_tensor.shape)}"
        )

    if before_tensor.shape[1] != 4:
        raise ValueError(
            f"Expected 4 channels, "
            f"got {before_tensor.shape[1]}"
        )

    if not torch.isfinite(
        before_tensor
    ).all():
        raise ValueError(
            "Before tensor contains non-finite values."
        )

    if not torch.isfinite(
        after_tensor
    ).all():
        raise ValueError(
            "After tensor contains non-finite values."
        )

    if device is None:
        target_device = next(
            model.parameters()
        ).device
    else:
        target_device = torch.device(device)

    before_tensor = before_tensor.to(
        target_device
    )

    after_tensor = after_tensor.to(
        target_device
    )

    with torch.no_grad():
        logits = model(
            before_tensor,
            after_tensor,
        )

        probability = torch.sigmoid(
            logits
        )

        mask = (
            probability >= threshold
        ).to(torch.uint8)

    probability_np = (
        probability.squeeze(1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    mask_np = (
        mask.squeeze(1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return probability_np, mask_np


if __name__ == "__main__":
    checkpoint = (
        "outputs/checkpoints/"
        "change_unet_synthetic_dev.pt"
    )

    model, device = load_change_model(
        checkpoint
    )

    before = np.random.default_rng(
        1
    ).random(
        (4, 64, 64),
        dtype=np.float32,
    )

    after = before.copy()

    after[
        :,
        20:40,
        20:40,
    ] *= 1.4

    probability, mask = predict_change_mask(
        model,
        before,
        after,
        device=device,
        threshold=0.5,
    )

    print(
        "=== CHANGE MODEL INFERENCE TEST ==="
    )

    print("Device:", device)

    print(
        "Probability shape:",
        probability.shape,
    )

    print(
        "Mask shape:",
        mask.shape,
    )

    print(
        "Probability range:",
        float(probability.min()),
        float(probability.max()),
    )

    print(
        "Mask dtype:",
        mask.dtype,
    )

    print(
        "Changed pixels:",
        int(mask.sum()),
    )

    print(
        "Finite probability:",
        bool(np.isfinite(probability).all()),
    )

    print(
        "\nSTEP 22G.07: PASS"
    )
