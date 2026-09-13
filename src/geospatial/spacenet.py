from __future__ import annotations

import csv
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

import rasterio
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform


@dataclass(frozen=True)
class SpaceNetBuilding:
    """Ground-truth building annotation from SpaceNet."""

    building_id: str
    pixel_geometry: Polygon
    geographic_geometry: Polygon


def load_spacenet_buildings(
    image_path: str,
    labels_archive: str,
    labels_member: str,
) -> list[SpaceNetBuilding]:
    """Load SpaceNet building polygons associated with a raster image."""

    raster_path = Path(image_path)
    archive_path = Path(labels_archive)

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster image does not exist: {raster_path}"
        )

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Labels archive does not exist: {archive_path}"
        )

    image_id = raster_path.stem

    if image_id.startswith("Pan-Sharpen_"):
        image_id = image_id[len("Pan-Sharpen_"):]

    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            raise ValueError("Raster must contain a valid CRS.")

        transform = raster.transform
        raster_bounds = raster.bounds

        with tarfile.open(archive_path, "r:gz") as tar:
            member = tar.getmember(labels_member)
            extracted = tar.extractfile(member)

            if extracted is None:
                raise ValueError(
                    f"Could not read label member: {labels_member}"
                )

            csv_text = extracted.read().decode("utf-8")

        reader = csv.DictReader(io.StringIO(csv_text))
        buildings: list[SpaceNetBuilding] = []

        for row in reader:
            if row["ImageId"] != image_id:
                continue

            pixel_geometry = wkt.loads(row["PolygonWKT_Pix"])

            if not isinstance(pixel_geometry, Polygon):
                raise ValueError(
                    f"Building {row['BuildingId']} is not a Polygon."
                )

            geographic_geometry = shapely_transform(
                lambda x, y, z=None: transform @ (x, y),
                pixel_geometry,
            )

            if not geographic_geometry.is_valid:
                geographic_geometry = geographic_geometry.buffer(0)

            min_x, min_y, max_x, max_y = geographic_geometry.bounds

            if not (
                min_x >= raster_bounds.left
                and max_x <= raster_bounds.right
                and min_y >= raster_bounds.bottom
                and max_y <= raster_bounds.top
            ):
                raise ValueError(
                    f"Building {row['BuildingId']} falls outside "
                    "the raster bounds."
                )

            buildings.append(
                SpaceNetBuilding(
                    building_id=row["BuildingId"],
                    pixel_geometry=pixel_geometry,
                    geographic_geometry=geographic_geometry,
                )
            )

    return buildings
