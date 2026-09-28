import pytest
from shapely.geometry import Point, box, Polygon
from pyproj import CRS
from src.schemas import Evidence
from src.executor.gis_executor import GISEvidenceExecutor
from src.evidence import EvidenceRegistry

def create_evidence(geometry, crs, e_id="E1"):
    return Evidence(
        evidence_id=e_id,
        source="test",
        task="task",
        model="m",
        modality="geospatial",
        confidence=1.0,
        geometry={"type": "Polygon", "coordinates": []} if geometry is None else geometry.__geo_interface__,
        metadata={"crs": crs}
    )

def test_geographic_buffer_reprojection():
    """Test EPSG:4326 -> projected CRS -> 500m buffer."""
    reg = EvidenceRegistry()
    executor = GISEvidenceExecutor(reg)
    
    geom = Point(0, 0).buffer(0.001)
    ev = create_evidence(geom, "EPSG:4326")
    reg.add(ev)
    
    res = executor.execute("buffer", parameters={"distance_m": 500.0}, source_evidence_ids=["E1"])
    assert res.measurement["operation"] == "buffer"
    assert res.measurement["crs"] == "EPSG:4326"
    assert res.measurement["projected_crs_used"] is not None
    assert "EPSG:32631" in res.measurement["projected_crs_used"]

def test_already_projected_input():
    """Test buffer on already projected input doesn't reproject."""
    reg = EvidenceRegistry()
    executor = GISEvidenceExecutor(reg)
    
    geom = Point(500000, 0).buffer(100)
    ev = create_evidence(geom, "EPSG:32631")
    reg.add(ev)
    
    res = executor.execute("buffer", parameters={"distance_m": 500.0}, source_evidence_ids=["E1"])
    assert res.measurement["projected_crs_used"] is None
    assert res.measurement["crs"] == "EPSG:32631"

def test_crs_mismatch():
    """Test CRS mismatch between two projected CRS."""
    reg = EvidenceRegistry()
    executor = GISEvidenceExecutor(reg)
    
    ev1 = create_evidence(Point(0, 0).buffer(1), "EPSG:32631", "E1")
    ev2 = create_evidence(Point(0, 0).buffer(1), "EPSG:32632", "E2")
    reg.add(ev1)
    reg.add(ev2)
    
    with pytest.raises(ValueError, match="requires matching CRS"):
        executor.execute("intersection", source_evidence_ids=["E1", "E2"])

def test_missing_crs():
    reg = EvidenceRegistry()
    executor = GISEvidenceExecutor(reg)
    
    ev = create_evidence(Point(0, 0).buffer(1), None, "E1")
    ev.metadata = {}
    reg.add(ev)
    
    with pytest.raises(ValueError, match="requires CRS metadata"):
        executor.execute("buffer", parameters={"distance_m": 500.0}, source_evidence_ids=["E1"])

def test_invalid_geometry():
    reg = EvidenceRegistry()
    executor = GISEvidenceExecutor(reg)
    
    ev = create_evidence(None, "EPSG:4326", "E1")
    reg.add(ev)
    
    with pytest.raises(ValueError, match="requires a projected CRS|contains no geometry"):
        executor.execute("buffer", parameters={"distance_m": 500.0}, source_evidence_ids=["E1"])

def test_geometry_outside_supported_crs_range():
    reg = EvidenceRegistry()
    executor = GISEvidenceExecutor(reg)
    
    geom = Point(200, 100).buffer(1) # Out of bounds for WGS84
    ev = create_evidence(geom, "EPSG:4326")
    reg.add(ev)
    
    with pytest.raises(ValueError, match="requires a projected CRS"):
        executor.execute("buffer", parameters={"distance_m": 500.0}, source_evidence_ids=["E1"])

def test_reprojection_round_trip():
    """Test buffer distance preservation and reprojection accuracy."""
    reg = EvidenceRegistry()
    executor = GISEvidenceExecutor(reg)
    
    # 0,0 is at the equator. 1 degree ~ 111km. So 0.001 degree ~ 111m.
    geom = Point(0, 0).buffer(0.001)
    ev = create_evidence(geom, "EPSG:4326")
    reg.add(ev)
    
    res = executor.execute("buffer", parameters={"distance_m": 100.0}, source_evidence_ids=["E1"])
    # Original shape radius is ~ 111m. Added 100m. Total radius ~ 211m. Area should be ~ pi * r^2.
    assert res.measurement["area_m2"] > 0
    assert "EPSG:4326" in res.metadata["crs"]
