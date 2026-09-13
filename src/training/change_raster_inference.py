from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from src.training.change_model_inference import (
    load_change_model,
    predict_change_mask,
)


PATCH_SIZE = 64
STRIDE = 64


def normalize_patch(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)

    output = np.zeros_like(
        image,
        dtype=np.float32,
    )

    for band_index in range(image.shape[0]):
        band = image[band_index]

        low = float(np.percentile(band, 2))
        high = float(np.percentile(band, 98))

        if high <= low:
            output[band_index] = 0.0
        else:
            output[band_index] = np.clip(
                (band - low) / (high - low),
                0.0,
                1.0,
            )

    return output


def predict_change_raster(
    model,
    before_path: str,
    after_path: str,
    *,
    device=None,
    threshold: float = 0.5,
) -> dict:
    before_path = Path(before_path)
    after_path = Path(after_path)

    if not before_path.exists():
        raise FileNotFoundError(
            f"Before raster does not exist: {before_path}"
        )

    if not after_path.exists():
        raise FileNotFoundError(
            f"After raster does not exist: {after_path}"
        )

    with rasterio.open(before_path) as before_ds:
        before_data = before_ds.read()

        before_shape = (
            before_ds.height,
            before_ds.width,
        )

        before_crs = before_ds.crs
        before_transform = before_ds.transform

    with rasterio.open(after_path) as after_ds:
        after_data = after_ds.read()

        after_shape = (
            after_ds.height,
            after_ds.width,
        )

        after_crs = after_ds.crs
        after_transform = after_ds.transform

    if before_shape != after_shape:
        raise ValueError(
            "Before/after raster dimensions differ: "
            f"{before_shape} vs {after_shape}"
        )

    if before_data.shape != after_data.shape:
        raise ValueError(
            "Before/after band shapes differ: "
            f"{before_data.shape} vs {after_data.shape}"
        )

    if before_data.shape[0] != 4:
        raise ValueError(
            f"Expected 4 bands, "
            f"found {before_data.shape[0]}"
        )

    if before_crs is None or after_crs is None:
        raise ValueError(
            "Both rasters must have valid CRS."
        )

    if before_crs != after_crs:
        raise ValueError(
            "Before/after CRS differ: "
            f"{before_crs} vs {after_crs}"
        )

    if before_transform != after_transform:
        raise ValueError(
            "Before/after transforms differ."
        )

    height, width = before_shape

    probability = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    covered_height = (
        (height // STRIDE) * STRIDE
    )

    covered_width = (
        (width // STRIDE) * STRIDE
    )

    for top in range(
        0,
        covered_height,
        STRIDE,
    ):
        for left in range(
            0,
            covered_width,
            STRIDE,
        ):

            before_patch = before_data[
                :,
                top:top + PATCH_SIZE,
                left:left + PATCH_SIZE,
            ]

            after_patch = after_data[
                :,
                top:top + PATCH_SIZE,
                left:left + PATCH_SIZE,
            ]

            before_patch = normalize_patch(
                before_patch
            )

            after_patch = normalize_patch(
                after_patch
            )

            patch_probability, patch_mask = (
                predict_change_mask(
                    model,
                    before_patch,
                    after_patch,
                    device=device,
                    threshold=threshold,
                )
            )

            probability[
                top:top + PATCH_SIZE,
                left:left + PATCH_SIZE,
            ] = patch_probability[0]

            mask[
                top:top + PATCH_SIZE,
                left:left + PATCH_SIZE,
            ] = patch_mask[0]

    return {
        "probability": probability,
        "mask": mask,
        "transform": before_transform,
        "crs": before_crs,
        "width": width,
        "height": height,
        "covered_width": covered_width,
        "covered_height": covered_height,
        "before": str(before_path.resolve()),
        "after": str(after_path.resolve()),
    }


if __name__ == "__main__":
    checkpoint = (
        "outputs/checkpoints/"
        "change_unet_synthetic_dev.pt"
    )

    # Development test: use the same scene for both dates.
    # A real temporal pair will be supplied later.
    image = (
        "data/remote_sensing/spacenet4/"
        "Pan-Sharpen_Atlanta_nadir53_catid_"
        "1030010003CD4300_743501_3721539.tif"
    )

    model, device = load_change_model(
        checkpoint
    )

    result = predict_change_raster(
        model,
        image,
        image,
        device=device,
    )

    print("=== CHANGE RASTER INFERENCE ===")
    print("Device:", device)
    print("Probability shape:",
          result["probability"].shape)
    print("Mask shape:",
          result["mask"].shape)
    print("Probability range:",
          float(result["probability"].min()),
          float(result["probability"].max()))
    print("Changed pixels:",
          int(result["mask"].sum()))
    print("CRS:",
          result["crs"])
    print("Covered:",
          result["covered_width"],
          "x",
          result["covered_height"])

    print("\nSTEP 22G.08: PASS")
