"""
test_hero_reasoning_e2e.py — Unit tests for hero multi-step geographic reasoning workflow.

Verifies:
  - Query decomposition of complex natural-language geographic requests
  - Required subtask identification (temporal, flood, building, 500m buffer)
  - Strict rejection of missing building model checkpoint weights (preventing fake output)
  - Rejection of missing temporal evidence
  - 500 m GIS buffer calculation correctness
  - Spatial intersection correctness
  - GeoReasonVerifier enforcement of scientific honesty rules
"""
import pytest
import os
import sys
from pathlib import Path
from shapely.geometry import Polygon, Point, box

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.controller.task_controller import TaskController
from src.planner.evidence_planner import EvidencePlanner
from src.executor.building_specialist import BuildingDetectionSpecialist
from src.executor.gis_executor import GISEvidenceExecutor
from src.evidence.registry import EvidenceRegistry
from src.schemas.evidence import Evidence
from src.verifier.georeason_verifier import GeoReasonVerifier
from src.geospatial.geometry import geometry_buffer, geometry_intersection, geometry_area


# ── 1. Query Decomposition Tests ─────────────────────────────

def test_hero_query_decomposition_task_controller():
    """Verify TaskController parses complex hero geographic query correctly."""
    tc = TaskController()
    spec = tc.build_task_spec("Find newly constructed buildings within 500 m of flooded areas.", 2)

    assert spec.task_type == "temporal_analysis"
    assert "temporal_analysis" in spec.required_capabilities
    assert "building_detection" in spec.required_capabilities
    assert "flood_detection" in spec.required_capabilities
    assert "buffer" in spec.spatial_operations
    assert spec.parameters.get("distance_m") == 500.0


def test_hero_query_evidence_plan_decomposition():
    """Verify EvidencePlanner builds a multi-step plan with all required subtasks."""
    tc = TaskController()
    spec = tc.build_task_spec("Find newly constructed buildings within 500 m of flooded areas.", 2)

    planner = EvidencePlanner()
    plan = planner.create_plan(spec)

    step_tasks = [step.task for step in plan.steps]
    step_ops = [step.operation for step in plan.steps]

    assert "temporal_analysis" in step_tasks
    assert "flood_detection" in step_tasks
    assert "building_detection" in step_tasks
    assert "buffer" in step_ops
    assert "verification" in step_ops


# ── 2. Building Model Checkpoint Rejection Test ──────────────

def test_building_specialist_rejects_missing_checkpoint(tmp_path):
    """Verify BuildingDetectionSpecialist fails honestly when model weights are missing."""
    fake_image = tmp_path / "test.tif"
    import rasterio
    from affine import Affine
    import numpy as np

    with rasterio.open(
        fake_image,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="uint8",
        crs="EPSG:32645",
        transform=Affine.identity(),
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=np.uint8))

    specialist = BuildingDetectionSpecialist(checkpoint_path=str(tmp_path / "nonexistent_building_model.pt"))

    with pytest.raises(FileNotFoundError, match="Building model checkpoint does not exist"):
        specialist.infer(inputs=[str(fake_image)])


# ── 3. GIS Buffer & Spatial Reasoning Correctness ────────────

def test_500m_buffer_correctness():
    """Verify 500 m buffer generation on spatial geometry."""
    poly = box(0, 0, 100, 100)  # 100x100m polygon
    buffered = geometry_buffer(poly, 500.0)

    assert buffered.is_valid
    # Buffer of 500m around 100x100 box expands bounds by 500 in all directions
    minx, miny, maxx, maxy = buffered.bounds
    assert pytest.approx(minx, abs=1.0) == -500.0
    assert pytest.approx(miny, abs=1.0) == -500.0
    assert pytest.approx(maxx, abs=1.0) == 600.0
    assert pytest.approx(maxy, abs=1.0) == 600.0


def test_spatial_intersection_correctness():
    """Verify spatial intersection between building buffer and flood polygon."""
    flood_poly = box(0, 0, 200, 200)
    building_poly = box(400, 400, 450, 450)  # Center is (425, 425), distance to flood is ~282m (<500m)

    building_buffer = geometry_buffer(building_poly, 500.0)
    inter = geometry_intersection(building_buffer, flood_poly)

    assert not inter.is_empty
    assert geometry_area(inter) > 0.0


def test_spatial_disjoint_outside_buffer():
    """Verify spatial intersection is empty when building is beyond buffer distance."""
    flood_poly = box(0, 0, 100, 100)
    building_poly = box(1000, 1000, 1050, 1050)  # Distance ~1272m (>500m)

    building_buffer = geometry_buffer(building_poly, 500.0)
    inter = geometry_intersection(building_buffer, flood_poly)

    assert inter.is_empty


# ── 4. Verification & Scientific Honesty Tests ────────────────

def test_verifier_rejects_missing_temporal_metadata():
    """Verify GeoReasonVerifier rejects temporal change evidence lacking T1/T2 timestamps."""
    ev = Evidence(
        evidence_id="CHANGE_invalid",
        source="ChangeSpecialist",
        task="temporal_analysis",
        model="BiTemporalChange",
        sensor="Sentinel-2",
        modality="optical",
        timestamp=None,
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
        measurement={},
        result={},
        confidence=0.7,
        provenance={},  # missing t1_timestamp / t2_timestamp
        metadata={},
    )
    verifier = GeoReasonVerifier(minimum_confidence=0.5)
    res = verifier.verify([ev], expected_task="temporal_analysis")

    assert res.verified is False
    assert any("missing T1 or T2 timestamp" in r for r in res.reasons)


def test_verifier_rejects_missing_geometry_for_change_task():
    """Verify GeoReasonVerifier rejects temporal change evidence without spatial geometry."""
    ev = Evidence(
        evidence_id="CHANGE_no_geom",
        source="ChangeSpecialist",
        task="temporal_analysis",
        model="BiTemporalChange",
        sensor="Sentinel-2",
        modality="optical",
        timestamp="2024-10-13",
        t1_timestamp="2023-10-22",
        t2_timestamp="2024-10-13",
        geometry=None,  # missing geometry
        measurement={},
        result={},
        confidence=0.7,
        provenance={},
        metadata={},
    )
    verifier = GeoReasonVerifier(minimum_confidence=0.5)
    res = verifier.verify([ev], expected_task="temporal_analysis")

    assert res.verified is False
    assert any("lacks valid geometry" in r for r in res.reasons)
