import numpy as np
import rasterio
from rasterio.transform import from_origin

path = "data/samples/test.tif"

data = np.zeros((4, 10, 10), dtype=np.uint16)

transform = from_origin(
    500000,
    2100000,
    10,
    10,
)

profile = {
    "driver": "GTiff",
    "height": 10,
    "width": 10,
    "count": 4,
    "dtype": "uint16",
    "crs": "EPSG:32643",
    "transform": transform,
}

with rasterio.open(path, "w", **profile) as dst:
    dst.write(data)

print(f"Synthetic GeoTIFF created: {path}")
