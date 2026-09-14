from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.executor.sar_specialist import SARSpecialist


def _write_sar(
    path: Path,
    *,
    values: np.ndarray,
    crs: str = "EPSG:32616",
    nodata: float | None = None,
) -> None:
    transform = from_origin(
        743501.0,
        3721989.0,
        0.5,
        0.5,
    )

    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }

    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(
            values.astype(np.float32),
            1,
        )


def test_sar_specialist_returns_standardized_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sar.tif"

    values = np.array(
        [
            [-10.0, -8.0, -6.0],
            [-5.0, -4.0, -3.0],
            [-2.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )

    _write_sar(path, values=values)

    evidence = SARSpecialist().infer(
        [str(path)],
        parameters={
            "sensor": "test_sar",
            "polarization": "VV",
            "band": "C",
        },
    )

    assert evidence.task == "sar_analysis"
    assert evidence.model == "SAR_Deterministic_Characterizer"
    assert evidence.modality == "sar"
    assert evidence.sensor == "test_sar"

    assert evidence.confidence == pytest.approx(0.80)

    assert evidence.result["analysis_type"] == (
        "sar_characterization"
    )
    assert evidence.result["crs"] == "EPSG:32616"

    assert evidence.measurement is not None
    assert evidence.measurement["valid_pixel_count"] == 9
    assert evidence.measurement["valid_fraction"] == pytest.approx(1.0)
    assert evidence.measurement["polarization"] == "VV"
    assert evidence.measurement["band"] == "C"

    assert evidence.provenance["perception_status"] == "CONNECTED"
    assert evidence.metadata["deterministic"] is True


def test_sar_specialist_handles_nodata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sar_nodata.tif"

    values = np.array(
        [
            [-10.0, -8.0],
            [-9999.0, -2.0],
        ],
        dtype=np.float32,
    )

    _write_sar(
        path,
        values=values,
        nodata=-9999.0,
    )

    evidence = SARSpecialist().infer(
        [str(path)]
    )

    assert evidence.measurement is not None
    assert evidence.measurement["valid_pixel_count"] == 3
    assert evidence.measurement["valid_fraction"] == pytest.approx(0.75)


def test_sar_specialist_rejects_missing_input() -> None:
    with pytest.raises(ValueError, match="requires at least one"):
        SARSpecialist().infer([])


def test_sar_specialist_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        SARSpecialist().infer(
            ["/tmp/does-not-exist-sar.tif"]
        )


def test_sar_specialist_rejects_multiband_raster(
    tmp_path: Path,
) -> None:
    path = tmp_path / "multiband.tif"

    transform = from_origin(
        743501.0,
        3721989.0,
        0.5,
        0.5,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=2,
        dtype="float32",
        crs="EPSG:32616",
        transform=transform,
    ) as dst:
        dst.write(
            np.ones(
                (2, 4, 4),
                dtype=np.float32,
            )
        )

    with pytest.raises(
        ValueError,
        match="single-band",
    ):
        SARSpecialist().infer(
            [str(path)]
        )


def test_sar_specialist_rejects_missing_crs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "no_crs.tif"

    transform = from_origin(
        0.0,
        10.0,
        1.0,
        1.0,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(
            np.ones(
                (1, 2, 2),
                dtype=np.float32,
            )
        )

    with pytest.raises(
        ValueError,
        match="valid CRS",
    ):
        SARSpecialist().infer(
            [str(path)]
        )
