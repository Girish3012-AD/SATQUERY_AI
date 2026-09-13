from affine import Affine
import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.executor.building_specialist import BuildingDetectionSpecialist


def create_test_raster(path):
    data = np.ones((10, 10), dtype=np.uint16)

    transform = from_origin(
        500000,
        2000,
        10,
        10,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=data.dtype,
        crs="EPSG:32643",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_building_specialist_capability():
    specialist = BuildingDetectionSpecialist()

    assert specialist.capability == "building_detection"


def test_building_specialist_requires_input(tmp_path):
    specialist = BuildingDetectionSpecialist()

    try:
        specialist.infer([])
    except ValueError as exc:
        assert "raster input" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")


def test_building_specialist_requires_existing_raster(tmp_path):
    specialist = BuildingDetectionSpecialist()

    missing = tmp_path / "missing.tif"

    try:
        specialist.infer([str(missing)])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_building_specialist_with_injected_detection(
    tmp_path,
):
    raster_path = tmp_path / "building_test.tif"
    create_test_raster(raster_path)

    specialist = BuildingDetectionSpecialist()

    evidence = specialist.infer(
        [str(raster_path)],
        parameters={
            "detected_boxes": [
                (1, 1, 4, 4, 0.90),
                (5, 5, 8, 8, 0.80),
            ]
        },
    )

    assert evidence.task == "building_detection"
    assert evidence.modality == "optical"

    assert evidence.result["building_count"] == 2
    assert evidence.result["has_detections"] is True

    assert evidence.geometry is not None
    assert evidence.geometry["type"] == "MultiPolygon"
