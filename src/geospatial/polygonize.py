"""
polygonize.py — Convert a binary raster mask to georeferenced Shapely polygons.

Uses rasterio.features.shapes() (deterministic).
Reuses existing geometry utilities from src.geospatial.geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

import rasterio.features

from .geometry import geometry_area


@dataclass
class PolygonResult:
    """Georeferenced polygons derived from a raster mask."""

    polygons: list[BaseGeometry]
    crs: str
    total_area_m2: float
    polygon_count: int
    pixel_count: int
    resolution_m: tuple[float, float]
    provenance: dict[str, Any]


def mask_to_polygons(
    mask: np.ndarray,
    transform: Affine,
    crs: str,
    *,
    resolution_m: tuple[float, float] = (10.0, 10.0),
    min_area_px: int = 4,
    connectivity: int = 4,
) -> PolygonResult:
    """
    Convert a boolean mask into georeferenced Shapely polygons.

    Parameters
    ----------
    mask : bool ndarray (H, W)
    transform : rasterio Affine transform for the mask raster
    crs : CRS string of the mask raster
    resolution_m : pixel resolution in metres (x, y)
    min_area_px : minimum polygon size in pixels (removes noise)
    connectivity : 4 or 8 pixel connectivity

    Returns
    -------
    PolygonResult with polygons in the raster's native CRS.
    """
    mask_uint8 = mask.astype(np.uint8)

    pixel_area_m2 = resolution_m[0] * resolution_m[1]

    polygons: list[BaseGeometry] = []
    pixel_count = int(np.sum(mask))

    shapes = rasterio.features.shapes(
        mask_uint8,
        mask=mask_uint8,
        connectivity=connectivity,
        transform=transform,
    )

    for geom_dict, value in shapes:
        if value != 1:
            continue

        poly = shape(geom_dict)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
            continue

        area_m2 = geometry_area(poly)
        if area_m2 < min_area_px * pixel_area_m2:
            continue

        polygons.append(poly)

    total_area = sum(geometry_area(p) for p in polygons)

    return PolygonResult(
        polygons=polygons,
        crs=crs,
        total_area_m2=float(total_area),
        polygon_count=len(polygons),
        pixel_count=pixel_count,
        resolution_m=resolution_m,
        provenance={
            "method": "rasterio.features.shapes",
            "connectivity": connectivity,
            "min_area_px": min_area_px,
            "pixel_area_m2": pixel_area_m2,
        },
    )
