from pathlib import Path

import pytest
from shapely.geometry import box

from src.geospatial import (
    bounds_to_geometry,
    buffer_geometry,
    calculate_area,
    calculate_distance,
    get_raster_metadata,
    intersect_geometries,
    validate_crs,
)


TEST_RASTER = Path("data/samples/test.tif")


def test_raster_metadata():
    metadata = get_raster_metadata(str(TEST_RASTER))

    assert metadata["driver"] == "GTiff"
    assert metadata["width"] == 10
    assert metadata["height"] == 10
    assert metadata["band_count"] == 4
    assert metadata["dtype"] == "uint16"
    assert metadata["crs"] == "EPSG:32643"
    assert metadata["resolution"] == (10.0, 10.0)


def test_raster_crs_is_valid():
    assert validate_crs(str(TEST_RASTER)) is True


def test_bounds_to_geometry():
    geometry = bounds_to_geometry((0, 0, 100, 100))

    assert geometry.is_valid
    assert calculate_area(geometry) == 10000.0


def test_intersection():
    first = box(0, 0, 100, 100)
    second = box(50, 50, 150, 150)

    result = intersect_geometries(first, second)

    assert not result.is_empty
    assert calculate_area(result) == 2500.0


def test_distance_between_overlapping_geometries():
    first = box(0, 0, 100, 100)
    second = box(50, 50, 150, 150)

    assert calculate_distance(first, second) == 0.0


def test_buffer():
    geometry = box(0, 0, 100, 100)

    buffered = buffer_geometry(geometry, 10)

    assert buffered.area > geometry.area


def test_negative_buffer_rejected():
    geometry = box(0, 0, 100, 100)

    with pytest.raises(ValueError, match="cannot be negative"):
        buffer_geometry(geometry, -10)


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError, match="Invalid bounds"):
        bounds_to_geometry((100, 100, 0, 0))


def test_500_meter_buffer_intersection():
    flooded = box(0, 0, 100, 100)
    building = box(400, 40, 450, 90)

    flooded_buffer = buffer_geometry(flooded, 500)
    result = intersect_geometries(flooded_buffer, building)

    assert not result.is_empty
    assert calculate_area(result) > 0
