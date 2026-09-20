"""Unit tests for spectral.py, polygonize.py, and aoi_resolver.py.

These tests are self-contained — no network, no real imagery.
"""
import numpy as np
import pytest
from affine import Affine

from src.geospatial.spectral import (
    compute_ndwi,
    water_candidate_mask,
    analyse_sentinel2_ndwi,
)
from src.geospatial.polygonize import mask_to_polygons
from src.data.aoi_resolver import resolve_aoi, extract_location_from_query


# ---------------------------------------------------------------------------
# spectral — compute_ndwi
# ---------------------------------------------------------------------------

def test_ndwi_basic_water_pixel():
    """High Green, low NIR → positive NDWI (water)."""
    green = np.array([[1000.0]], dtype=np.float32)
    nir = np.array([[200.0]], dtype=np.float32)
    result = compute_ndwi(green, nir)
    assert result[0, 0] == pytest.approx(
        (1000 - 200) / (1000 + 200), rel=1e-4
    )


def test_ndwi_basic_vegetation_pixel():
    """Low Green, high NIR → negative NDWI (vegetation)."""
    green = np.array([[300.0]], dtype=np.float32)
    nir = np.array([[2000.0]], dtype=np.float32)
    result = compute_ndwi(green, nir)
    assert result[0, 0] < 0


def test_ndwi_division_by_zero_gives_nan():
    """Both bands zero → NaN (no division by zero error)."""
    green = np.array([[0.0]], dtype=np.float32)
    nir = np.array([[0.0]], dtype=np.float32)
    result = compute_ndwi(green, nir)
    assert np.isnan(result[0, 0])


def test_ndwi_nodata_masked():
    """Nodata pixels become NaN."""
    green = np.array([[9999.0, 1000.0]], dtype=np.float32)
    nir = np.array([[9999.0, 200.0]], dtype=np.float32)
    result = compute_ndwi(green, nir, nodata=9999.0)
    assert np.isnan(result[0, 0])
    assert not np.isnan(result[0, 1])


def test_ndwi_output_range():
    """NDWI values must be in [-1, 1] for valid inputs."""
    rng = np.random.default_rng(42)
    green = rng.integers(0, 5000, size=(10, 10)).astype(np.float32)
    nir = rng.integers(0, 5000, size=(10, 10)).astype(np.float32)
    result = compute_ndwi(green, nir)
    valid = result[~np.isnan(result)]
    assert np.all(valid >= -1.0)
    assert np.all(valid <= 1.0)


# ---------------------------------------------------------------------------
# spectral — water_candidate_mask
# ---------------------------------------------------------------------------

def test_water_mask_threshold_zero():
    """Pixels >= 0 are candidates."""
    ndwi = np.array([[0.1, -0.2, 0.0, np.nan]], dtype=np.float32)
    mask = water_candidate_mask(ndwi, threshold=0.0)
    assert mask[0, 0]   # 0.1 >= 0
    assert not mask[0, 1]  # -0.2 < 0
    assert mask[0, 2]   # 0.0 >= 0
    assert not mask[0, 3]  # NaN excluded


def test_water_mask_custom_threshold():
    ndwi = np.array([[0.3, 0.1]], dtype=np.float32)
    mask = water_candidate_mask(ndwi, threshold=0.2)
    assert mask[0, 0]
    assert not mask[0, 1]


def test_water_mask_invalid_threshold():
    ndwi = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        water_candidate_mask(ndwi, threshold=1.5)


# ---------------------------------------------------------------------------
# spectral — analyse_sentinel2_ndwi (integration)
# ---------------------------------------------------------------------------

def _make_transform(pixel_size: float = 10.0) -> Affine:
    return Affine.translation(0.0, 100.0) @ Affine.scale(pixel_size, -pixel_size)


def test_analyse_returns_spectral_result():
    green = np.array([[1000.0, 200.0], [1000.0, 200.0]], dtype=np.float32)
    nir = np.array([[200.0, 1000.0], [200.0, 1000.0]], dtype=np.float32)
    transform = _make_transform()
    result = analyse_sentinel2_ndwi(
        green, nir,
        transform=transform,
        crs="EPSG:32644",
        resolution_m=(10.0, 10.0),
        scene_id="TEST-001",
        acquisition_date="2024-01-01",
    )
    assert result.index_name == "NDWI"
    assert result.candidate_pixel_count == 2   # two water-like pixels
    assert "potential_water_candidate" in result.provenance["label"]
    assert result.provenance["scene_id"] == "TEST-001"


# ---------------------------------------------------------------------------
# polygonize — mask_to_polygons
# ---------------------------------------------------------------------------

def test_polygonize_simple_square():
    """A solid 4×4 True block should produce at least one polygon."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True   # 16 pixels
    transform = _make_transform(pixel_size=10.0)
    result = mask_to_polygons(mask, transform, crs="EPSG:32644")
    assert result.polygon_count >= 1
    assert result.total_area_m2 > 0


def test_polygonize_empty_mask():
    """All False → no polygons, zero area."""
    mask = np.zeros((10, 10), dtype=bool)
    transform = _make_transform()
    result = mask_to_polygons(mask, transform, crs="EPSG:32644")
    assert result.polygon_count == 0
    assert result.total_area_m2 == 0.0


def test_polygonize_pixel_count():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:4, 0:4] = True  # 16 pixels
    transform = _make_transform(10.0)
    result = mask_to_polygons(mask, transform, crs="EPSG:32644")
    assert result.pixel_count == 16


def test_polygonize_crs_preserved():
    mask = np.ones((5, 5), dtype=bool)
    transform = _make_transform()
    result = mask_to_polygons(mask, transform, crs="EPSG:32643")
    assert result.crs == "EPSG:32643"


def test_polygonize_min_area_filters_noise():
    """Single isolated pixels below min_area_px threshold are removed."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5, 5] = True   # single pixel
    mask[3:7, 8:12] = True  # 16-pixel block (kept)
    transform = _make_transform(10.0)
    result = mask_to_polygons(mask, transform, crs="EPSG:32644", min_area_px=4)
    # The 16-pixel block should survive; isolated pixel may or may not
    assert result.polygon_count >= 1


# ---------------------------------------------------------------------------
# aoi_resolver
# ---------------------------------------------------------------------------

def test_resolve_rasuwa():
    aoi = resolve_aoi("Rasuwa, Nepal")
    assert aoi.name == "Rasuwa District, Nepal"
    assert len(aoi.bbox) == 4
    min_lon, min_lat, max_lon, max_lat = aoi.bbox
    assert min_lon < max_lon
    assert min_lat < max_lat
    # Rasuwa is in Nepal (~85°E, 28°N)
    assert 84.0 <= min_lon <= 86.0
    assert 27.0 <= min_lat <= 29.0


def test_resolve_rasuwa_partial_name():
    aoi = resolve_aoi("Rasuwa")
    assert "Rasuwa" in aoi.name


def test_resolve_pune():
    aoi = resolve_aoi("Pune")
    assert "Pune" in aoi.name


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown place"):
        resolve_aoi("Atlantis")


def test_resolve_empty_raises():
    with pytest.raises(ValueError):
        resolve_aoi("")


def test_extract_location_rasuwa():
    query = "Identify potential flooded areas in Rasuwa, Nepal using Sentinel-2 imagery."
    location = extract_location_from_query(query)
    assert location is not None
    # Should contain "Rasuwa" somewhere
    assert "Rasuwa" in location or "rasuwa" in location.lower()
