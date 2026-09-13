import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.executor.change_specialist import ChangeSpecialist


def create_raster(
    path,
    data,
    crs="EPSG:32643",
    transform=None,
):
    if transform is None:
        transform = from_origin(
            500000,
            2000,
            10,
            10,
        )

    data = np.asarray(data)

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dataset:
        dataset.write(data)


def test_change_specialist_capability():
    specialist = ChangeSpecialist()

    assert specialist.capability == "temporal_analysis"


def test_change_requires_exactly_two_inputs(tmp_path):
    specialist = ChangeSpecialist()

    with pytest.raises(
        ValueError,
        match="exactly two",
    ):
        specialist.infer([str(tmp_path / "one.tif")])


def test_change_rejects_missing_before_image(tmp_path):
    specialist = ChangeSpecialist()

    after = tmp_path / "after.tif"

    create_raster(
        after,
        np.ones((10, 10), dtype=np.uint16),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Before raster",
    ):
        specialist.infer(
            [
                str(tmp_path / "before.tif"),
                str(after),
            ]
        )


def test_change_detects_modified_pixels(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"

    before_data = np.zeros(
        (10, 10),
        dtype=np.uint16,
    )

    after_data = before_data.copy()
    after_data[0:5, 0:5] = 1000

    create_raster(before, before_data)
    create_raster(after, after_data)

    specialist = ChangeSpecialist(
        default_threshold=0.10,
    )

    evidence = specialist.infer(
        [
            str(before),
            str(after),
        ]
    )

    assert evidence.task == "temporal_analysis"
    assert evidence.modality == "optical"
    assert evidence.result["changed"] is True
    assert evidence.result["changed_pixels"] > 0
    assert evidence.result["change_percentage"] > 0
    assert evidence.confidence > 0


def test_change_detects_no_change(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"

    data = np.arange(
        100,
        dtype=np.uint16,
    ).reshape(10, 10)

    create_raster(before, data)
    create_raster(after, data.copy())

    specialist = ChangeSpecialist()

    evidence = specialist.infer(
        [
            str(before),
            str(after),
        ]
    )

    assert evidence.result["changed"] is False
    assert evidence.result["changed_pixels"] == 0
    assert evidence.result["change_percentage"] == 0.0


def test_change_rejects_dimension_mismatch(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"

    create_raster(
        before,
        np.ones((10, 10), dtype=np.uint16),
    )

    create_raster(
        after,
        np.ones((8, 10), dtype=np.uint16),
    )

    specialist = ChangeSpecialist()

    with pytest.raises(
        ValueError,
        match="identical dimensions",
    ):
        specialist.infer(
            [
                str(before),
                str(after),
            ]
        )


def test_change_rejects_crs_mismatch(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"

    data = np.ones(
        (10, 10),
        dtype=np.uint16,
    )

    create_raster(
        before,
        data,
        crs="EPSG:32643",
    )

    create_raster(
        after,
        data,
        crs="EPSG:4326",
    )

    specialist = ChangeSpecialist()

    with pytest.raises(
        ValueError,
        match="same CRS",
    ):
        specialist.infer(
            [
                str(before),
                str(after),
            ]
        )


def test_change_rejects_invalid_threshold(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"

    data = np.ones(
        (10, 10),
        dtype=np.uint16,
    )

    create_raster(before, data)
    create_raster(after, data)

    specialist = ChangeSpecialist()

    with pytest.raises(
        ValueError,
        match="threshold",
    ):
        specialist.infer(
            [
                str(before),
                str(after),
            ],
            parameters={"threshold": 0},
        )


def test_change_supports_multiple_bands(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"

    before_data = np.zeros(
        (3, 10, 10),
        dtype=np.uint16,
    )

    after_data = before_data.copy()
    after_data[:, 2:6, 2:6] = 1000

    create_raster(before, before_data)
    create_raster(after, after_data)

    specialist = ChangeSpecialist(
        default_threshold=0.10,
    )

    evidence = specialist.infer(
        [
            str(before),
            str(after),
        ]
    )

    assert evidence.result["changed"] is True
    assert evidence.metadata["band_count"] == 3
