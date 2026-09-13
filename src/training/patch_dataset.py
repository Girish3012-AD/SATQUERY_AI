from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert uint16 multispectral image data to float32 [0, 1].

    Normalization is performed independently per band using
    the 2nd and 98th percentiles.
    """
    image = image.astype(np.float32)

    output = np.zeros_like(image, dtype=np.float32)

    for band_index in range(image.shape[0]):
        band = image[band_index]

        valid = np.isfinite(band) & (band > 0)

        if not np.any(valid):
            continue

        values = band[valid]

        p2, p98 = np.percentile(values, (2, 98))

        if p98 <= p2:
            output[band_index] = np.clip(
                band / max(float(p98), 1.0),
                0.0,
                1.0,
            )
        else:
            output[band_index] = np.clip(
                (band - p2) / (p98 - p2),
                0.0,
                1.0,
            )

    return output


def create_patch_dataset(
    image_path: str,
    mask_path: str,
    output_dir: str,
    patch_size: int = 64,
    stride: int = 64,
) -> dict[str, Any]:
    """
    Create image/mask patches aligned to the source raster grid.

    Each image patch is stored as:
        .npy -> shape (bands, patch_size, patch_size)

    Each mask patch is stored as:
        .npy -> shape (patch_size, patch_size)

    Only complete patches are generated.
    """

    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    if stride <= 0:
        raise ValueError("stride must be positive.")

    image = Path(image_path)
    mask = Path(mask_path)
    output = Path(output_dir)

    if not image.exists():
        raise FileNotFoundError(f"Image does not exist: {image}")

    if not mask.exists():
        raise FileNotFoundError(f"Mask does not exist: {mask}")

    image_output = output / "images"
    mask_output = output / "masks"

    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image) as image_src:
        image_data = image_src.read()

        image_height = image_src.height
        image_width = image_src.width
        image_transform = image_src.transform
        image_crs = image_src.crs

        with rasterio.open(mask) as mask_src:
            if mask_src.count != 1:
                raise ValueError(
                    "Building mask must contain exactly one band."
                )

            if mask_src.width != image_width:
                raise ValueError("Image and mask widths do not match.")

            if mask_src.height != image_height:
                raise ValueError("Image and mask heights do not match.")

            if mask_src.transform != image_transform:
                raise ValueError(
                    "Image and mask transforms do not match."
                )

            if mask_src.crs != image_crs:
                raise ValueError(
                    "Image and mask CRS values do not match."
                )

            mask_data = mask_src.read(1)

    unique_mask_values = set(np.unique(mask_data).tolist())

    if not unique_mask_values.issubset({0, 1}):
        raise ValueError(
            f"Mask must contain only 0 and 1. "
            f"Found: {sorted(unique_mask_values)}"
        )

    normalized_image = _normalize_image(image_data)

    patch_records: list[dict[str, Any]] = []

    patch_index = 0

    for row in range(0, image_height - patch_size + 1, stride):
        for col in range(0, image_width - patch_size + 1, stride):

            image_patch = normalized_image[
                :,
                row:row + patch_size,
                col:col + patch_size,
            ]

            mask_patch = mask_data[
                row:row + patch_size,
                col:col + patch_size,
            ]

            image_filename = f"patch_{patch_index:05d}.npy"
            mask_filename = f"patch_{patch_index:05d}.npy"

            np.save(image_output / image_filename, image_patch)
            np.save(mask_output / mask_filename, mask_patch.astype(np.uint8))

            building_pixels = int(np.count_nonzero(mask_patch))
            total_pixels = int(mask_patch.size)

            patch_records.append(
                {
                    "patch_id": patch_index,
                    "image": str(
                        image_output / image_filename
                    ),
                    "mask": str(
                        mask_output / mask_filename
                    ),
                    "row": row,
                    "col": col,
                    "building_pixels": building_pixels,
                    "total_pixels": total_pixels,
                    "building_fraction": (
                        building_pixels / total_pixels
                    ),
                }
            )

            patch_index += 1

    manifest_path = output / "manifest.npy"
    np.save(
        manifest_path,
        np.array(patch_records, dtype=object),
        allow_pickle=True,
    )

    building_patch_count = sum(
        record["building_pixels"] > 0
        for record in patch_records
    )

    return {
        "image": str(image),
        "mask": str(mask),
        "output_dir": str(output),
        "patch_size": patch_size,
        "stride": stride,
        "image_shape": tuple(image_data.shape),
        "mask_shape": tuple(mask_data.shape),
        "patch_count": len(patch_records),
        "patches_with_buildings": building_patch_count,
        "patches_without_buildings": (
            len(patch_records) - building_patch_count
        ),
        "manifest": str(manifest_path),
    }
