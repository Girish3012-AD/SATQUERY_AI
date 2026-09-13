from __future__ import annotations

from typing import Tuple

from affine import Affine
from shapely.geometry import Polygon


def pixel_bbox_to_geographic_polygon(
    bbox: Tuple[float, float, float, float],
    transform: Affine,
) -> Polygon:
    """
    Convert a pixel-space bounding box into a polygon in the raster's
    native projected coordinate system.

    bbox:
        (x_min, y_min, x_max, y_max)

    Pixel coordinates use:
        x = column
        y = row

    transform:
        Rasterio affine transform associated with the GeoTIFF.
    """
    if len(bbox) != 4:
        raise ValueError(
            "Bounding box must contain exactly four values: "
            "(x_min, y_min, x_max, y_max)."
        )

    x_min, y_min, x_max, y_max = map(float, bbox)

    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")

    if y_min >= y_max:
        raise ValueError("y_min must be smaller than y_max.")

    # Transform all four pixel corners into the raster's CRS.
    top_left = transform * (x_min, y_min)
    top_right = transform * (x_max, y_min)
    bottom_right = transform * (x_max, y_max)
    bottom_left = transform * (x_min, y_max)

    polygon = Polygon(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ]
    )

    if not polygon.is_valid:
        raise ValueError("Generated geographic polygon is invalid.")

    return polygon
