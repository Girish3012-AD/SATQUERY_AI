from __future__ import annotations

from pathlib import Path

import rasterio


def validate_optical_sar_alignment(
    optical_path: str,
    sar_path: str,
) -> dict:

    optical = Path(optical_path)
    sar = Path(sar_path)

    if not optical.exists():
        raise FileNotFoundError(
            f"Optical raster does not exist: {optical}"
        )

    if not sar.exists():
        raise FileNotFoundError(
            f"SAR raster does not exist: {sar}"
        )

    with rasterio.open(optical) as optical_ds:
        optical_crs = optical_ds.crs
        optical_transform = optical_ds.transform
        optical_width = optical_ds.width
        optical_height = optical_ds.height
        optical_bounds = optical_ds.bounds
        optical_res = optical_ds.res
        optical_count = optical_ds.count

    with rasterio.open(sar) as sar_ds:
        sar_crs = sar_ds.crs
        sar_transform = sar_ds.transform
        sar_width = sar_ds.width
        sar_height = sar_ds.height
        sar_bounds = sar_ds.bounds
        sar_res = sar_ds.res
        sar_count = sar_ds.count

    same_crs = optical_crs == sar_crs
    same_shape = (
        optical_width == sar_width
        and optical_height == sar_height
    )
    same_transform = (
        optical_transform == sar_transform
    )
    same_bounds = (
        optical_bounds == sar_bounds
    )
    same_resolution = (
        optical_res == sar_res
    )

    aligned = all(
        [
            same_crs,
            same_shape,
            same_transform,
            same_bounds,
            same_resolution,
        ]
    )

    return {
        "aligned": aligned,
        "same_crs": same_crs,
        "same_shape": same_shape,
        "same_transform": same_transform,
        "same_bounds": same_bounds,
        "same_resolution": same_resolution,
        "optical_bands": optical_count,
        "sar_bands": sar_count,
        "optical_crs": str(optical_crs),
        "sar_crs": str(sar_crs),
    }
