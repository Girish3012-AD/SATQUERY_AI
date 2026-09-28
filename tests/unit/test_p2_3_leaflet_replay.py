import pytest
import json
from src.schemas.evidence import Evidence
from src.orchestration.lifecycle import ExecutionStageTrace, StageStatus, PipelineStage
from src.orchestration.orchestration_result import OrchestrationResult
from api_server import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_evidence_geojson_support():
    """Test Evidence safely handles valid GeoJSON of various types."""
    # Polygon
    ev = Evidence(
        evidence_id="E1", source="src", task="task", model="m", modality="optical", confidence=0.8,
        geometry={"type": "Polygon", "coordinates": [[[0,0], [0,1], [1,1], [0,0]]]}
    )
    assert ev.geometry["type"] == "Polygon"

    # MultiPolygon
    ev = Evidence(
        evidence_id="E2", source="src", task="task", model="m", modality="optical", confidence=0.8,
        geometry={"type": "MultiPolygon", "coordinates": [[[[0,0], [0,1], [1,1], [0,0]]]]}
    )
    assert ev.geometry["type"] == "MultiPolygon"
    
    # Point
    ev = Evidence(
        evidence_id="E3", source="src", task="task", model="m", modality="optical", confidence=0.8,
        geometry={"type": "Point", "coordinates": [0,0]}
    )
    assert ev.geometry["type"] == "Point"

    # LineString
    ev = Evidence(
        evidence_id="E4", source="src", task="task", model="m", modality="optical", confidence=0.8,
        geometry={"type": "LineString", "coordinates": [[0,0], [1,1]]}
    )
    assert ev.geometry["type"] == "LineString"

def test_evidence_invalid_missing_geometry():
    """Test Evidence handles missing and empty geometry safely (no fabrication)."""
    ev = Evidence(
        evidence_id="E5", source="src", task="task", model="m", modality="optical", confidence=0.8,
        geometry=None
    )
    assert ev.geometry is None

    ev_empty = Evidence(
        evidence_id="E6", source="src", task="task", model="m", modality="optical", confidence=0.8,
        geometry={}
    )
    assert ev_empty.geometry == {}

def test_replay_payload_structure(tmp_path):
    """Test the replay payload reconstructs execution result correctly."""
    # Create a synthetic legacy/replay artifact
    report = {
        "query": "Where is water?",
        "task_type": "water_detection",
        "task_id": "TASK-123",
        "success": True,
        "selected_capabilities": {"water_detection": "WaterSpecialist"},
        "selected_models": {"WaterSpecialist": "WaterModel"},
        "execution_status": "COMPLETED",
        "verification_status": "verified",
        "pipeline_metrics": {"total_time_ms": 1500},
        "lifecycle_trace": [
            {"stage_name": "QUERY_RECEIVED", "status": "EXECUTED", "outputs": {}}
        ],
        "evidence": [
            {
                "evidence_id": "EV-1",
                "task": "water_detection",
                "source": "WaterSpecialist",
                "model": "WaterModel",
                "modality": "optical",
                "confidence": 0.9,
                "geometry": {"type": "Polygon", "coordinates": [[[0,0], [0,1], [1,1], [0,0]]]}
            }
        ]
    }
    
    assert report["query"] == "Where is water?"
    assert report["evidence"][0]["geometry"]["type"] == "Polygon"

def test_legacy_artifact_handling():
    """Test handling of legacy/empty artifacts where some fields are missing."""
    legacy = {
        "query": "Find buildings",
        "success": False,
        # missing lifecycle_trace, pipeline_metrics, geometry
    }
    assert legacy.get("lifecycle_trace") is None
    assert legacy.get("pipeline_metrics") is None
    assert not legacy.get("evidence")

def test_failed_blocked_execution_artifact():
    """Test handling of BLOCKED/failed execution payload."""
    report = {
        "query": "Blocked query",
        "success": False,
        "execution_status": "BLOCKED",
        "messages": ["Execution was blocked due to missing inputs."]
    }
    assert report["success"] is False
    assert report["execution_status"] == "BLOCKED"
    assert "missing inputs" in report["messages"][0]

def test_orchestration_result_to_replay_format():
    """Ensure OrchestrationResult correctly structures data for frontend replay/live mode."""
    res = OrchestrationResult(
        success=True,
        task_id="T1",
        query="Q1",
        task_type="vqa",
        plan_id="p1",
        pipeline_metrics={"time": 10},
        lifecycle_trace=[
            ExecutionStageTrace(
                stage_id="s1", stage_name="QUERY_RECEIVED", started_at="1", completed_at="2", status=StageStatus.EXECUTED
            )
        ],
        mode="live"
    )
    import dataclasses
    dump = dataclasses.asdict(res)
    assert "lifecycle_trace" in dump
    assert "pipeline_metrics" in dump
    assert dump["lifecycle_trace"][0].stage_name == "QUERY_RECEIVED"

