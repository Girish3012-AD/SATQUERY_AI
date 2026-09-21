"""Focused unit tests for the spatial grounding E2E path."""
import pytest
import numpy as np
import rasterio
from affine import Affine
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.controller.task_controller import TaskController
from src.executor.water_specialist import WaterSpecialist
from src.schemas.evidence import Evidence
from src.verifier.georeason_verifier import GeoReasonVerifier


# ── A: Grounding Query Routing ─────────────────────────────
def test_grounding_query_routing():
    controller = TaskController()
    spec = controller.build_task_spec(
        query="Highlight the water body referred to in the image.",
        input_count=1,
    )
    assert "water_detection" in spec.required_capabilities
    assert spec.task_type == "specialized_analysis"


def test_grounding_water_query_routing():
    controller = TaskController()
    spec = controller.build_task_spec(
        query="Locate the water bodies in the area.",
        input_count=1,
    )
    assert "water_detection" in spec.required_capabilities


# ── B: Missing Image Rejection ──────────────────────────────
def test_grounding_rejects_missing_image():
    spec = WaterSpecialist()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        spec.infer(inputs=["non_existent_b03.tif", "non_existent_b08.tif"])


# ── C: Invalid/Empty Input Rejection ───────────────────────
def test_grounding_rejects_empty_inputs():
    spec = WaterSpecialist()
    with pytest.raises(ValueError, match="at least one raster input"):
        spec.infer(inputs=[])


# ── D: Spatial Geometry Requirement ─────────────────────────
def test_grounding_produces_spatial_geometry(tmp_path):
    # Generate synthetic 10x10 rasters with NDWI > 0 water region
    b03_path = tmp_path / "b03.tif"
    b08_path = tmp_path / "b08.tif"

    transform = Affine.translation(1000, 2000) * Affine.scale(10, -10)
    crs = "EPSG:32645"

    b03_data = np.ones((1, 10, 10), dtype=np.float32) * 0.8
    b08_data = np.ones((1, 10, 10), dtype=np.float32) * 0.2

    for path, data in [(b03_path, b03_data), (b08_path, b08_data)]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)

    spec = WaterSpecialist(ndwi_threshold=0.0)
    ev = spec.infer(inputs=[str(b03_path), str(b08_path)])

    assert ev.geometry is not None
    assert ev.geometry["type"] in {"Polygon", "MultiPolygon"}
    assert "bbox_geographic" in ev.result
    assert ev.result["bbox_geographic"] is not None


# ── E: Invalid Geometry Rejection in Verifier ──────────────
def test_verifier_rejects_missing_geometry_for_spatial_task():
    ev = Evidence(
        evidence_id="GROUNDING_WATER_invalid",
        source="WaterSpecialist",
        task="water_detection",
        model="NDWI_Sentinel2_WaterGrounding",
        sensor="Sentinel-2",
        modality="optical",
        timestamp="2024-10-13",
        geometry=None,  # Missing geometry
        measurement={},
        result={},
        confidence=0.7,
        provenance={},
        metadata={},
    )
    verifier = GeoReasonVerifier(minimum_confidence=0.5)
    res = verifier.verify([ev], expected_task="spatial_analysis")
    assert res.verified is False
    assert res.status == "abstain"


# ── F: Evidence Schema Compliance ───────────────────────────
def test_grounding_evidence_schema_compliance(tmp_path):
    b03_path = tmp_path / "b03.tif"
    b08_path = tmp_path / "b08.tif"

    transform = Affine.translation(0, 0) * Affine.scale(10, -10)
    crs = "EPSG:4326"

    b03 = np.full((1, 10, 10), 0.5, dtype=np.float32)
    b08 = np.full((1, 10, 10), 0.1, dtype=np.float32)

    for path, data in [(b03_path, b03), (b08_path, b08)]:
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32", crs=crs, transform=transform
        ) as dst:
            dst.write(data)

    spec = WaterSpecialist()
    ev = spec.infer(inputs=[str(b03_path), str(b08_path)])

    assert ev.task == "water_detection"
    assert ev.modality == "optical"
    assert ev.sensor == "Sentinel-2"
    assert "water_polygon_count" in ev.measurement
    assert "total_water_area_km2" in ev.measurement


# ── G: Provenance Requirement ──────────────────────────────
def test_grounding_provenance_requirement(tmp_path):
    b03_path = tmp_path / "b03.tif"
    b08_path = tmp_path / "b08.tif"

    transform = Affine.translation(0, 0) * Affine.scale(10, -10)
    crs = "EPSG:32645"

    b03 = np.full((1, 10, 10), 0.5, dtype=np.float32)
    b08 = np.full((1, 10, 10), 0.1, dtype=np.float32)

    for path, data in [(b03_path, b03), (b08_path, b08)]:
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32", crs=crs, transform=transform
        ) as dst:
            dst.write(data)

    spec = WaterSpecialist()
    ev = spec.infer(inputs=[str(b03_path), str(b08_path)])

    assert "index" in ev.provenance
    assert ev.provenance["index"] == "NDWI"
    assert "source_path" in ev.provenance
    assert "analysis_type" in ev.provenance


# ── H: Verifier Accepts Valid Grounding Evidence ────────────
def test_verifier_accepts_valid_grounding_evidence(tmp_path):
    b03_path = tmp_path / "b03.tif"
    b08_path = tmp_path / "b08.tif"

    transform = Affine.translation(0, 0) * Affine.scale(10, -10)
    crs = "EPSG:32645"

    b03 = np.full((1, 10, 10), 0.6, dtype=np.float32)
    b08 = np.full((1, 10, 10), 0.1, dtype=np.float32)

    for path, data in [(b03_path, b03), (b08_path, b08)]:
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32", crs=crs, transform=transform
        ) as dst:
            dst.write(data)

    spec = WaterSpecialist()
    ev = spec.infer(inputs=[str(b03_path), str(b08_path)])

    verifier = GeoReasonVerifier(minimum_confidence=0.5)
    res = verifier.verify([ev], expected_task="water_detection")

    assert res.verified is True
    assert res.status == "verified"


# ── I: Verifier Rejects Low Confidence Grounding Evidence ──
def test_verifier_rejects_low_confidence_grounding():
    ev = Evidence(
        evidence_id="GROUNDING_WATER_low_conf",
        source="WaterSpecialist",
        task="water_detection",
        model="NDWI_Sentinel2_WaterGrounding",
        sensor="Sentinel-2",
        modality="optical",
        timestamp="2024-10-13",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        measurement={},
        result={},
        confidence=0.2,  # Too low
        provenance={},
        metadata={},
    )
    verifier = GeoReasonVerifier(minimum_confidence=0.5)
    res = verifier.verify([ev], expected_task="water_detection")

    assert res.verified is False


# ── J: Deterministic Evidence ID ───────────────────────────
def test_grounding_evidence_id_deterministic(tmp_path):
    b03_path = tmp_path / "b03.tif"
    b08_path = tmp_path / "b08.tif"

    transform = Affine.translation(0, 0) * Affine.scale(10, -10)
    crs = "EPSG:32645"

    b03 = np.full((1, 10, 10), 0.6, dtype=np.float32)
    b08 = np.full((1, 10, 10), 0.1, dtype=np.float32)

    for path, data in [(b03_path, b03), (b08_path, b08)]:
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32", crs=crs, transform=transform
        ) as dst:
            dst.write(data)

    spec = WaterSpecialist()
    ev1 = spec.infer(inputs=[str(b03_path), str(b08_path)], parameters={"scene_id": "scene_A"})
    ev2 = spec.infer(inputs=[str(b03_path), str(b08_path)], parameters={"scene_id": "scene_A"})

    assert ev1.evidence_id == ev2.evidence_id
    assert ev1.evidence_id.startswith("GROUNDING_WATER_")
