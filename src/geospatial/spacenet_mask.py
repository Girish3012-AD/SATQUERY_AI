from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize

from src.geospatial.spacenet import load_spacenet_buildings


def create_spacenet_building_mask(
    image_path: str,
    labels_archive: str,
    labels_member: str,
    output_path: str,
) -> dict:
    """
    Rasterize SpaceNet building polygons onto the source image grid.

    Output:
        Single-band uint8 GeoTIFF where:
            0 = background
            1 = building
    """

    image = Path(image_path)
    output = Path(output_path)

    buildings = load_spacenet_buildings(
        image_path=image_path,
        labels_archive=labels_archive,
        labels_member=labels_member,
    )

    if not buildings:
        raise ValueError(
            f"No SpaceNet buildings found for image: {image}"
        )

    with rasterio.open(image) as src:
        shapes = [
            (building.geographic_geometry, 1)
            for building in buildings
        ]

        mask = rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )

        output.parent.mkdir(parents=True, exist_ok=True)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=None,
            compress="lzw",
        )

        with rasterio.open(output, "w", **profile) as dst:
            dst.write(mask, 1)

        building_pixels = int(np.count_nonzero(mask))
        total_pixels = int(mask.size)

        return {
            "image": str(image),
            "mask": str(output),
            "width": src.width,
            "height": src.height,
            "building_count": len(buildings),
            "building_pixels": building_pixels,
            "total_pixels": total_pixels,
            "building_fraction": building_pixels / total_pixels,
            "crs": src.crs.to_string(),
            "resolution": src.res,
        }
