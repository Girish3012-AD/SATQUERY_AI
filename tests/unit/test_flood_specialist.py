"""
Unit tests for FloodSpecialist.

Uses small synthetic GeoTIFF rasters — no network, no real imagery.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from src.executor.flood_specialist import FloodSpecialist
from src.schemas.evidence import Evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_multiband_tiff(
    path: Path,
    bands: list[np.ndarray],
    crs: str = "EPSG:32644",
    pixel_size: float = 10.0,
) -> None:
    """Write a multi-band float32 GeoTIFF for testing."""
    h, w = bands[0].shape
    transform = Affine.translation(0.0, h * pixel_size) @ Affine.scale(
        pixel_size, -pixel_size
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=len(bands),
        dtype=np.float32,
        crs=CRS.from_string(crs),
        transform=transform,
    ) as ds:
        for i, band in enumerate(bands, start=1):
            ds.write(band.astype(np.float32), i)


def _write_single_band_tiff(
    path: Path,
    data: np.ndarray,
    crs: str = "EPSG:32644",
    pixel_size: float = 10.0,
) -> None:
    _write_multiband_tiff(path, [data], crs=crs, pixel_size=pixel_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_flood_specialist_capability():
    spec = FloodSpecialist()
    assert spec.capability == "flood_detection"


def test_flood_specialist_model_name():
    assert FloodSpecialist.MODEL_NAME == "NDWI_Sentinel2_FloodCandidate"


def test_flood_specialist_two_single_band_inputs():
    """Two single-band GeoTIFFs (B03 + B08) → Evidence."""
    h, w = 20, 20
    # Water pixels: high green, low NIR
    green = np.full((h, w), 2000.0, dtype=np.float32)
    nir = np.full((h, w), 300.0, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        b03 = Path(tmp) / "b03.tif"
        b08 = Path(tmp) / "b08.tif"
        _write_single_band_tiff(b03, green)
        _write_single_band_tiff(b08, nir)

        spec = FloodSpecialist(ndwi_threshold=0.0)
        evidence = spec.infer([str(b03), str(b08)])

    assert isinstance(evidence, Evidence)
    assert evidence.task == "flood_detection"
    assert evidence.measurement["candidate_pixel_count"] == h * w
    assert evidence.result["label"] == "potential_water_candidate"
    assert "NOT confirmed flooding" in evidence.result["note"]
    assert evidence.confidence <= 1.0
    assert evidence.geometry is not None   # largest polygon returned


def test_flood_specialist_multiband_input():
    """Single multi-band GeoTIFF (band 1 = B03, band 4 = B08)."""
    h, w = 10, 10
    green = np.full((h, w), 1800.0, dtype=np.float32)
    dummy_b2 = np.zeros((h, w), dtype=np.float32)
    dummy_b3 = np.zeros((h, w), dtype=np.float32)
    nir = np.full((h, w), 400.0, dtype=np.float32)  # band 4

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sentinel2_4band.tif"
        _write_multiband_tiff(path, [green, dummy_b2, dummy_b3, nir])

        spec = FloodSpecialist(ndwi_threshold=0.0)
        evidence = spec.infer([str(path)], parameters={
            "green_band": 1, "nir_band": 4
        })

    assert isinstance(evidence, Evidence)
    assert evidence.measurement["candidate_pixel_count"] > 0


def test_flood_specialist_no_water_pixels():
    """All land pixels (high NIR) → zero candidate count, no polygon."""
    h, w = 10, 10
    green = np.full((h, w), 300.0, dtype=np.float32)
    nir = np.full((h, w), 3000.0, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        b03 = Path(tmp) / "b03.tif"
        b08 = Path(tmp) / "b08.tif"
        _write_single_band_tiff(b03, green)
        _write_single_band_tiff(b08, nir)

        spec = FloodSpecialist(ndwi_threshold=0.0)
        evidence = spec.infer([str(b03), str(b08)])

    assert evidence.measurement["candidate_pixel_count"] == 0
    assert evidence.measurement["candidate_polygon_count"] == 0
    assert evidence.geometry is None


def test_flood_specialist_missing_input_raises():
    spec = FloodSpecialist()
    with pytest.raises(FileNotFoundError):
        spec.infer(["/nonexistent/b03.tif", "/nonexistent/b08.tif"])


def test_flood_specialist_empty_inputs_raises():
    spec = FloodSpecialist()
    with pytest.raises(ValueError, match="at least one raster"):
        spec.infer([])


def test_flood_specialist_evidence_id_deterministic():
    """Same inputs → same evidence_id."""
    h, w = 8, 8
    green = np.full((h, w), 1500.0, dtype=np.float32)
    nir = np.full((h, w), 500.0, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        b03 = Path(tmp) / "b03.tif"
        b08 = Path(tmp) / "b08.tif"
        _write_single_band_tiff(b03, green)
        _write_single_band_tiff(b08, nir)

        spec = FloodSpecialist()
        e1 = spec.infer([str(b03), str(b08)], {"scene_id": "SCENE-A"})
        e2 = spec.infer([str(b03), str(b08)], {"scene_id": "SCENE-A"})

    assert e1.evidence_id == e2.evidence_id


def test_flood_specialist_area_consistency():
    """Area should be approximately candidate_pixels * pixel_area."""
    h, w = 10, 10
    # 5×5 water block = 25 pixels at 10m resolution = 2500 m²
    green = np.zeros((h, w), dtype=np.float32)
    nir = np.zeros((h, w), dtype=np.float32)
    green[0:5, 0:5] = 2000.0
    nir[0:5, 0:5] = 300.0
    # rest: high NIR → land
    green[5:, :] = 300.0
    nir[5:, :] = 3000.0
    green[:, 5:] = 300.0
    nir[:, 5:] = 3000.0

    with tempfile.TemporaryDirectory() as tmp:
        b03 = Path(tmp) / "b03.tif"
        b08 = Path(tmp) / "b08.tif"
        _write_single_band_tiff(b03, green, pixel_size=10.0)
        _write_single_band_tiff(b08, nir, pixel_size=10.0)

        spec = FloodSpecialist(ndwi_threshold=0.0)
        evidence = spec.infer([str(b03), str(b08)])

    area_m2 = evidence.measurement["total_candidate_area_m2"]
    assert area_m2 == pytest.approx(2500.0, rel=0.05)
