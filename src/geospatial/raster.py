from pathlib import Path

import rasterio
from pyproj import CRS


def get_raster_metadata(image_path: str) -> dict:
    """Read essential geospatial metadata from a raster."""

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Raster does not exist: {path}")

    with rasterio.open(path) as dataset:
        crs = dataset.crs.to_string() if dataset.crs else None

        return {
            "path": str(path.resolve()),
            "driver": dataset.driver,
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": crs,
            "bounds": (
                dataset.bounds.left,
                dataset.bounds.bottom,
                dataset.bounds.right,
                dataset.bounds.top,
            ),
            "resolution": (
                abs(dataset.transform.a),
                abs(dataset.transform.e),
            ),
        }


def validate_crs(image_path: str) -> bool:
    """Return True when the raster has a valid CRS."""

    with rasterio.open(image_path) as dataset:
        if dataset.crs is None:
            return False

        CRS.from_user_input(dataset.crs)
        return True
