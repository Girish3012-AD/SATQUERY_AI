from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from shapely.geometry import shape


PATCH_SIZE = 64


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Apply training-time per-band 2nd/98th percentile normalization."""

    if patch.ndim != 3:
        raise ValueError(
            "Expected patch with shape [C,H,W]."
        )

    output = np.zeros_like(
        patch,
        dtype=np.float32,
    )

    for band_index in range(patch.shape[0]):
        band = patch[band_index].astype(
            np.float32,
            copy=False,
        )

        low = float(np.percentile(band, 2))
        high = float(np.percentile(band, 98))

        if high <= low:
            output[band_index] = 0.0
            continue

        normalized = (
            band - low
        ) / (
            high - low
        )

        output[band_index] = np.clip(
            normalized,
            0.0,
            1.0,
        )

    return output


def predict_building_raster(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device | None = None,
    threshold: float = 0.5,
    patch_size: int = PATCH_SIZE,
) -> dict[str, Any]:
    """
    Run BuildingUNet over a complete GeoTIFF using the training preprocessing.

    Returns:
        {
            "probability": HxW float32 array,
            "mask": HxW uint8 array,
            "transform": rasterio Affine,
            "crs": CRS,
            "width": int,
            "height": int,
            "covered_width": int,
            "covered_height": int,
        }
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input raster does not exist: {path}"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0.0 and 1.0."
        )

    if patch_size <= 0:
        raise ValueError(
            "patch_size must be positive."
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    with rasterio.open(path) as dataset:
        if dataset.count != 4:
            raise ValueError(
                f"BuildingUNet expects 4 bands, "
                f"but raster contains {dataset.count}."
            )

        if dataset.crs is None:
            raise ValueError(
                "Building detection requires a raster with a valid CRS."
            )

        image = dataset.read()
        transform = dataset.transform
        crs = dataset.crs
        width = dataset.width
        height = dataset.height

    covered_width = (
        width // patch_size
    ) * patch_size

    covered_height = (
        height // patch_size
    ) * patch_size

    probability = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for y in range(
            0,
            covered_height,
            patch_size,
        ):
            for x in range(
                0,
                covered_width,
                patch_size,
            ):
                patch = image[
                    :,
                    y:y + patch_size,
                    x:x + patch_size,
                ]

                patch = _normalize_patch(patch)

                tensor = torch.from_numpy(
                    patch
                ).unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                )

                logits = model(tensor)

                probabilities = torch.sigmoid(
                    logits
                )[0, 0].cpu().numpy()

                probability[
                    y:y + patch_size,
                    x:x + patch_size,
                ] = probabilities

    mask = (
        probability >= threshold
    ).astype(np.uint8)

    return {
        "probability": probability,
        "mask": mask,
        "transform": transform,
        "crs": crs,
        "width": width,
        "height": height,
        "covered_width": covered_width,
        "covered_height": covered_height,
    }


def polygonize_building_mask(
    mask: np.ndarray,
    transform: Any,
    min_area_m2: float = 4.0,
) -> list[dict[str, Any]]:
    """
    Convert a binary building mask into geographic polygons.

    Small regions below min_area_m2 are discarded.
    """

    if mask.ndim != 2:
        raise ValueError(
            "Building mask must have shape [H,W]."
        )

    if min_area_m2 < 0.0:
        raise ValueError(
            "min_area_m2 must be non-negative."
        )

    polygons: list[dict[str, Any]] = []

    for geometry, value in shapes(
        mask.astype(np.uint8),
        mask=mask.astype(bool),
        transform=transform,
    ):
        if int(value) != 1:
            continue

        polygon = shape(geometry)

        if polygon.is_empty:
            continue

        if polygon.area < min_area_m2:
            continue

        polygons.append(
            {
                "geometry": polygon,
                "area_m2": float(polygon.area),
            }
        )

    return polygons
