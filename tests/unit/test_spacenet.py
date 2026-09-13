from pathlib import Path

import pytest

from src.geospatial.spacenet import load_spacenet_buildings


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE = (
    PROJECT_ROOT
    / "data/remote_sensing/spacenet4/"
    "Pan-Sharpen_Atlanta_nadir53_catid_1030010003CD4300_"
    "743501_3721539.tif"
)

ARCHIVE = (
    PROJECT_ROOT
    / "data/remote_sensing/spacenet4/"
    "summaryData.tar.gz"
)

LABEL_MEMBER = (
    "summaryData/"
    "Atlanta_nadir53_catid_1030010003CD4300_Train.csv"
)


@pytest.mark.skipif(
    not IMAGE.exists() or not ARCHIVE.exists(),
    reason="SpaceNet sample data is not available",
)
def test_load_spacenet_buildings():
    buildings = load_spacenet_buildings(
        image_path=str(IMAGE),
        labels_archive=str(ARCHIVE),
        labels_member=LABEL_MEMBER,
    )

    assert len(buildings) == 59

    first = buildings[0]

    assert first.building_id == "0"
    assert first.pixel_geometry.is_valid
    assert first.geographic_geometry.is_valid
    assert first.geographic_geometry.area > 0


@pytest.mark.skipif(
    not IMAGE.exists() or not ARCHIVE.exists(),
    reason="SpaceNet sample data is not available",
)
def test_buildings_are_inside_raster():
    import rasterio

    buildings = load_spacenet_buildings(
        image_path=str(IMAGE),
        labels_archive=str(ARCHIVE),
        labels_member=LABEL_MEMBER,
    )

    with rasterio.open(IMAGE) as raster:
        raster_bounds = raster.bounds

        for building in buildings:
            min_x, min_y, max_x, max_y = (
                building.geographic_geometry.bounds
            )

            assert min_x >= raster_bounds.left
            assert max_x <= raster_bounds.right
            assert min_y >= raster_bounds.bottom
            assert max_y <= raster_bounds.top


def test_missing_image_rejected():
    with pytest.raises(FileNotFoundError):
        load_spacenet_buildings(
            image_path="/does/not/exist.tif",
            labels_archive=str(ARCHIVE),
            labels_member=LABEL_MEMBER,
        )


def test_missing_archive_rejected():
    with pytest.raises(FileNotFoundError):
        load_spacenet_buildings(
            image_path=str(IMAGE),
            labels_archive="/does/not/exist.tar.gz",
            labels_member=LABEL_MEMBER,
        )
