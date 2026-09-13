from pathlib import Path

import rasterio

from src.schemas.image import ImageRef


class ImageValidator:
    """Validate and inspect a remote-sensing raster image."""

    SUPPORTED_DRIVERS = {
        "GTiff",
        "JP2OpenJPEG",
        "HFA",
        "COG",
    }

    def validate(self, image_path: str) -> ImageRef:
        """
        Open a raster image, validate basic properties,
        and return a structured ImageRef.
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        try:
            with rasterio.open(path) as dataset:

                if dataset.driver not in self.SUPPORTED_DRIVERS:
                    raise ValueError(
                        f"Unsupported raster format: {dataset.driver}"
                    )

                bounds = (
                    dataset.bounds.left,
                    dataset.bounds.bottom,
                    dataset.bounds.right,
                    dataset.bounds.top,
                )

                crs = dataset.crs.to_string() if dataset.crs else None

                resolution_x = abs(dataset.transform.a)
                resolution_y = abs(dataset.transform.e)

                return ImageRef(
                    path=str(path.resolve()),
                    filename=path.name,
                    format=dataset.driver,
                    width=dataset.width,
                    height=dataset.height,
                    band_count=dataset.count,
                    dtype=str(dataset.dtypes[0]),
                    crs=crs,
                    resolution_x=resolution_x,
                    resolution_y=resolution_y,
                    bounds=bounds,
                    metadata=dict(dataset.tags()),
                )

        except rasterio.errors.RasterioIOError as exc:
            raise ValueError(
                f"Unable to read raster image: {path}"
            ) from exc
