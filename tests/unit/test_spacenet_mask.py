from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.geospatial.spacenet_mask import create_spacenet_building_mask


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
def test_create_building_mask(tmp_path):
    output = tmp_path / "building_mask.tif"

    result = create_spacenet_building_mask(
        image_path=str(IMAGE),
        labels_archive=str(ARCHIVE),
        labels_member=LABEL_MEMBER,
        output_path=str(output),
    )

    assert output.exists()
    assert result["building_count"] == 59
    assert result["width"] == 900
    assert result["height"] == 900
    assert result["building_pixels"] == 67683
    assert result["building_fraction"] > 0


@pytest.mark.skipif(
    not IMAGE.exists() or not ARCHIVE.exists(),
    reason="SpaceNet sample data is not available",
)
def test_mask_matches_source_grid(tmp_path):
    output = tmp_path / "building_mask.tif"

    create_spacenet_building_mask(
        image_path=str(IMAGE),
        labels_archive=str(ARCHIVE),
        labels_member=LABEL_MEMBER,
        output_path=str(output),
    )

    with rasterio.open(IMAGE) as image:
        with rasterio.open(output) as mask:
            assert mask.width == image.width
            assert mask.height == image.height
            assert mask.transform == image.transform
            assert mask.crs == image.crs
            assert mask.res == image.res
            assert mask.count == 1
            assert mask.dtypes[0] == "uint8"
            assert mask.nodata is None


@pytest.mark.skipif(
    not IMAGE.exists() or not ARCHIVE.exists(),
    reason="SpaceNet sample data is not available",
)
def test_mask_contains_only_binary_labels(tmp_path):
    output = tmp_path / "building_mask.tif"

    create_spacenet_building_mask(
        image_path=str(IMAGE),
        labels_archive=str(ARCHIVE),
        labels_member=LABEL_MEMBER,
        output_path=str(output),
    )

    with rasterio.open(output) as mask:
        data = mask.read(1)

    unique_values = set(np.unique(data).tolist())

    assert unique_values.issubset({0, 1})
    assert 0 in unique_values
    assert 1 in unique_values
