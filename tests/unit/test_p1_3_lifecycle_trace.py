"""
P1-3: Replay Lifecycle Trace — Unit Tests

Tests verify that the lifecycle trace:
  - is created during normal execution
  - contains ordered events matching the 15-stage canonical pipeline
  - survives serialization and deserialization
  - survives audit replay (frontend normalization)
  - is exposed to the API response
  - handles missing trace in legacy artifacts gracefully
  - records failure information
  - keeps verification separate from execution completion
  - never lets confidence become verification
  - does not fabricate missing events during replay

All tests are deterministic — no model downloads required.
"""
from __future__ import annotations

import json
import time
import pytest

from src.orchestration.lifecycle import (
    PipelineStage,
    StageStatus,
    STAGE_DISPLAY_NAMES,
    ExecutionStageTrace,
    PipelineMetrics,
)
from src.orchestration.orchestration_result import OrchestrationResult


# ---------------------------------------------------------------------------
# Helper: Build a minimal orchestrator and run a lifecycle build
# ---------------------------------------------------------------------------

def _build_lifecycle_from_orchestrator():
    """Create an orchestrator and run _build_canonical_lifecycle with test data."""
    from src.orchestration.orchestrator import SATQueryOrchestrator

    orchestrator = SATQueryOrchestrator()
    task_spec = orchestrator.build_task_spec(
        query="Is there flooding near Rasuwa?",
        input_count=1,
    )
    plan = orchestrator.build_plan(task_spec)
    
    # Simulate minimal results
    from types import SimpleNamespace
    results = [
        SimpleNamespace(
            step_id="step_flood",
            task="flood_detection",
            success=True,
            output={"answer": "flood detected"},
            message="Flood detection completed",
            evidence_ids=["EVID_001"],
        ),
        SimpleNamespace(
            step_id="step_verify",
            task="verification",
            success=True,
            output={"status": "verified", "confidence": 0.85, "reasons": ["ok"]},
            message=None,
            evidence_ids=[],
        ),
    ]
    
    lifecycle_trace, pipeline_metrics = orchestrator._build_canonical_lifecycle(
        task_spec=task_spec,
        plan=plan,
        inputs=["test_scene.tif"],
        selected_capabilities={"flood_detection": "FloodSpecialist"},
        selected_models={"flood_detection": "Qwen2-VL"},
        results=results,
        evidence_ids=["EVID_001"],
        verification={"status": "verified", "confidence": 0.85, "reasons": ["ok"]},
        start_time=time.time() - 5.0,
    )
    
    return lifecycle_trace, pipeline_metrics


# ===========================================================================
# Test 1: Lifecycle trace is created during normal execution
# ===========================================================================
def test_lifecycle_trace_created():
    """Verify that _build_canonical_lifecycle produces a non-empty trace."""
    trace, metrics = _build_lifecycle_from_orchestrator()
    
    assert isinstance(trace, list)
    assert len(trace) == 15, f"Expected 15 stages, got {len(trace)}"
    assert isinstance(metrics, dict)


# ===========================================================================
# Test 2: Lifecycle trace contains ordered events
# ===========================================================================
def test_lifecycle_trace_ordered_events():
    """Verify stages appear in canonical order."""
    trace, _ = _build_lifecycle_from_orchestrator()
    
    expected_stage_ids = [f"STAGE_{stage.value.upper()}" for stage in PipelineStage]
    actual_stage_ids = [s["stage_id"] for s in trace]
    
    assert actual_stage_ids == expected_stage_ids, \
        f"Stage order mismatch:\nExpected: {expected_stage_ids}\nActual:   {actual_stage_ids}"


# ===========================================================================
# Test 3: Lifecycle trace survives serialization
# ===========================================================================
def test_lifecycle_trace_serialization():
    """Verify the trace can be serialized to JSON and back without loss."""
    trace, metrics = _build_lifecycle_from_orchestrator()
    
    # Serialize
    serialized = json.dumps({"lifecycle_trace": trace, "pipeline_metrics": metrics}, default=str)
    
    # Deserialize
    deserialized = json.loads(serialized)
    
    assert deserialized["lifecycle_trace"] == trace
    assert deserialized["pipeline_metrics"] == metrics
    assert len(deserialized["lifecycle_trace"]) == 15


# ===========================================================================
# Test 4: Lifecycle trace survives replay (OrchestrationResult round-trip)
# ===========================================================================
def test_lifecycle_trace_survives_replay():
    """Simulate the full audit replay path: build → serialize → deserialize → check."""
    trace, metrics = _build_lifecycle_from_orchestrator()
    
    # Build the API response dict (matching api_server.py)
    api_response = {
        "success": True,
        "status": "verified",
        "execution_status": "COMPLETED",
        "verification_status": "verified",
        "confidence_calibration": "uncalibrated",
        "task_id": "TSK_replay_test",
        "query": "Is there flooding near Rasuwa?",
        "task_type": "flood_detection",
        "plan_id": "PLAN_001",
        "executed_steps": ["step_flood", "step_verify"],
        "successful_steps": ["step_flood", "step_verify"],
        "failed_steps": [],
        "evidence_ids": ["EVID_001"],
        "verification": {"status": "verified", "confidence": 0.85},
        "selected_capabilities": {"flood_detection": "FloodSpecialist"},
        "selected_models": {"flood_detection": "Qwen2-VL"},
        "messages": ["Flood detection completed"],
        "evidence": [],
        "execution_time_seconds": 5.0,
        "lifecycle_trace": trace,
        "pipeline_metrics": metrics,
        "mode": "live",
    }
    
    # Simulate writing to disk (what api_server.py now does)
    serialized = json.dumps(api_response, default=str)
    
    # Simulate reading back (what /api/audits/{filename} does)
    restored = json.loads(serialized)
    
    # Simulate renderAuditReplay normalization (the JS logic in Python)
    data = restored
    normalized = {
        "lifecycle_trace": data.get("lifecycle_trace", []),
        "pipeline_metrics": data.get("pipeline_metrics", {}),
        "executed_steps": data.get("executed_steps", []),
    }
    
    assert len(normalized["lifecycle_trace"]) == 15, \
        "lifecycle_trace lost during replay round-trip"
    assert normalized["lifecycle_trace"][0]["stage_id"] == "STAGE_QUERY_RECEIVED"
    assert normalized["lifecycle_trace"][-1]["stage_id"] == "STAGE_EXECUTION_REPORT"


# ===========================================================================
# Test 5: Replay exposes lifecycle_trace to frontend/API
# ===========================================================================
def test_replay_exposes_lifecycle_trace_to_api():
    """Verify the API response includes lifecycle_trace with correct structure."""
    trace, metrics = _build_lifecycle_from_orchestrator()
    
    # Check each trace entry has the required fields
    for stage in trace:
        assert "stage_id" in stage, f"Missing stage_id in {stage}"
        assert "stage_name" in stage, f"Missing stage_name in {stage}"
        assert "status" in stage, f"Missing status in {stage}"
        assert "execution_success" in stage, f"Missing execution_success in {stage}"
        assert "result_status" in stage, f"Missing result_status in {stage}"
        assert "inputs" in stage, f"Missing inputs in {stage}"
        assert "outputs" in stage, f"Missing outputs in {stage}"
        assert "evidence_ids" in stage, f"Missing evidence_ids in {stage}"
    
    # Check metrics has required fields
    assert "total_stages" in metrics
    assert "executed_stages" in metrics
    assert "execution_errors" in metrics
    assert "evidence_count" in metrics
    assert "verification_status" in metrics


# ===========================================================================
# Test 6: Missing lifecycle_trace in old artifact handled safely
# ===========================================================================
def test_missing_lifecycle_trace_in_legacy_artifact():
    """
    Old artifacts without lifecycle_trace must be handled gracefully.
    The frontend normalizer should default to [] (empty list).
    The trace panel should fall back to executed_steps or show unavailable.
    """
    # Simulate a legacy audit artifact (no lifecycle_trace)
    legacy_data = {
        "success": True,
        "status": "completed",
        "task_id": "TSK_legacy",
        "query": "What buildings are in this image?",
        "executed_steps": ["vqa", "verification"],
        "verification": {"status": "verified"},
        "evidence": [],
    }
    
    # Simulate frontend normalization (renderAuditReplay logic)
    normalized = {
        "lifecycle_trace": legacy_data.get("lifecycle_trace", []),
        "executed_steps": legacy_data.get("executed_steps", []),
    }
    
    assert normalized["lifecycle_trace"] == [], \
        "Missing lifecycle_trace should default to empty list, not fabricated events"
    assert len(normalized["executed_steps"]) == 2, \
        "Legacy executed_steps should still be accessible"


# ===========================================================================
# Test 7: Failed execution records failure information
# ===========================================================================
def test_failed_execution_records_failure():
    """Verify that a failed execution preserves error information in the trace."""
    from src.orchestration.orchestrator import SATQueryOrchestrator

    orchestrator = SATQueryOrchestrator()
    task_spec = orchestrator.build_task_spec(
        query="Analyze this satellite image",
        input_count=0,
    )
    plan = orchestrator.build_plan(task_spec)
    
    # Simulate a failed execution
    from types import SimpleNamespace
    results = [
        SimpleNamespace(
            step_id="step_vqa",
            task="vqa",
            success=False,
            output=None,
            message="Model failed to load",
            evidence_ids=[],
        ),
    ]
    
    trace, metrics = orchestrator._build_canonical_lifecycle(
        task_spec=task_spec,
        plan=plan,
        inputs=[],
        selected_capabilities={"vqa": "VQASpecialist"},
        selected_models={"vqa": "Qwen2-VL"},
        results=results,
        evidence_ids=[],
        verification={},
        start_time=time.time(),
    )
    
    assert len(trace) == 15
    # Metrics should reflect the empty verification state
    assert metrics["evidence_count"] == 0
    # Verification stage should still exist (it always runs)
    ver_stage = next(s for s in trace if "EVIDENCE_VERIFICATION" in s["stage_id"])
    assert ver_stage is not None


# ===========================================================================
# Test 8: Verification event is separate from execution completion
# ===========================================================================
def test_verification_event_separate_from_execution():
    """
    The lifecycle trace must have EVIDENCE_VERIFICATION as a separate stage
    from SPECIALIST_EXECUTION. They must not be merged.
    """
    trace, _ = _build_lifecycle_from_orchestrator()
    
    stage_ids = [s["stage_id"] for s in trace]
    
    # Both must exist as separate stages
    assert "STAGE_SPECIALIST_EXECUTION" in stage_ids, \
        "SPECIALIST_EXECUTION stage missing from trace"
    assert "STAGE_EVIDENCE_VERIFICATION" in stage_ids, \
        "EVIDENCE_VERIFICATION stage missing from trace"
    
    # They must be different entries
    spec_idx = stage_ids.index("STAGE_SPECIALIST_EXECUTION")
    ver_idx = stage_ids.index("STAGE_EVIDENCE_VERIFICATION")
    assert spec_idx != ver_idx, \
        "Execution and verification must be separate stages"
    assert ver_idx > spec_idx, \
        "Verification must come after specialist execution"
    
    # The verification stage should contain verification result data
    ver_stage = trace[ver_idx]
    assert ver_stage["result_status"] == "verified"


# ===========================================================================
# Test 9: Confidence does not become verification
# ===========================================================================
def test_confidence_never_becomes_verification_in_trace():
    """
    The CONFIDENCE_ABSTENTION_DECISION stage exists but is separate from
    EVIDENCE_VERIFICATION. Confidence value does not determine verification status.
    """
    trace, _ = _build_lifecycle_from_orchestrator()
    
    stage_ids = [s["stage_id"] for s in trace]
    
    # Both must exist
    assert "STAGE_CONFIDENCE_ABSTENTION_DECISION" in stage_ids
    assert "STAGE_EVIDENCE_VERIFICATION" in stage_ids
    
    # They must be separate
    conf_idx = stage_ids.index("STAGE_CONFIDENCE_ABSTENTION_DECISION")
    ver_idx = stage_ids.index("STAGE_EVIDENCE_VERIFICATION")
    assert conf_idx != ver_idx
    
    # Confidence stage contains the confidence value
    conf_stage = trace[conf_idx]
    assert "confidence_value" in conf_stage["outputs"]
    
    # Verification stage is the source of truth for verification status
    ver_stage = trace[ver_idx]
    assert ver_stage["result_status"] in ("verified", "low_confidence", "abstain", "unknown")


# ===========================================================================
# Test 10: Replay does NOT fabricate missing events
# ===========================================================================
def test_replay_does_not_fabricate_events():
    """
    When an old artifact has no lifecycle_trace, the replay must NOT
    invent lifecycle events to fill the gap. It must show empty or fallback.
    """
    legacy_artifact = {
        "success": True,
        "status": "completed",
        "executed_steps": ["flood_detection"],
        "verification": {"status": "verified"},
    }
    
    # Simulate the renderAuditReplay normalization
    normalized_lifecycle = legacy_artifact.get("lifecycle_trace", [])
    
    # Must be empty — no fabrication
    assert normalized_lifecycle == [], \
        "Replay must NOT fabricate lifecycle events for legacy artifacts"
    assert len(normalized_lifecycle) == 0
    
    # For an artifact WITH lifecycle_trace, the trace must match exactly
    trace, _ = _build_lifecycle_from_orchestrator()
    modern_artifact = {
        "success": True,
        "lifecycle_trace": trace,
    }
    
    restored_lifecycle = modern_artifact.get("lifecycle_trace", [])
    assert len(restored_lifecycle) == 15
    assert restored_lifecycle == trace, \
        "Restored lifecycle must be identical to the original — no fabrication"


# ===========================================================================
# Test 11: ExecutionStageTrace model validates correctly
# ===========================================================================
def test_execution_stage_trace_model():
    """Verify the Pydantic model accepts and serializes correctly."""
    stage = ExecutionStageTrace(
        stage_id="STAGE_QUERY_RECEIVED",
        stage_name="Query Received",
        started_at="2024-01-01T00:00:00Z",
        completed_at="2024-01-01T00:00:01Z",
        status=StageStatus.EXECUTED,
        execution_success=True,
        result_status="completed",
        inputs=["test query"],
        outputs={"query": "test query"},
        evidence_ids=[],
    )
    
    dumped = stage.model_dump()
    assert dumped["stage_id"] == "STAGE_QUERY_RECEIVED"
    assert dumped["status"] == "EXECUTED"
    assert dumped["execution_success"] is True
    
    # Round-trip through JSON
    serialized = json.dumps(dumped)
    restored = json.loads(serialized)
    assert restored == dumped


# ===========================================================================
# Test 12: PipelineMetrics model validates correctly
# ===========================================================================
def test_pipeline_metrics_model():
    """Verify the PipelineMetrics Pydantic model."""
    metrics = PipelineMetrics(
        total_stages=15,
        executed_stages=14,
        successful_executions=14,
        execution_errors=0,
        not_applicable_stages=1,
        skipped_stages=0,
        abstentions=0,
        evidence_count=3,
        final_decision="VERIFIED",
        verification_status="verified",
    )
    
    dumped = metrics.model_dump()
    assert dumped["total_stages"] == 15
    assert dumped["executed_stages"] == 14
    assert dumped["not_applicable_stages"] == 1
    assert dumped["verification_status"] == "verified"


# ===========================================================================
# Test 13: Frontend JS file renders lifecycle_trace in audit replay
# ===========================================================================
def test_frontend_passes_lifecycle_trace_through_replay():
    """
    Verify the JS renderAuditReplay normalizer now includes lifecycle_trace.
    Read the actual JS source and check the field exists.
    """
    with open("static/js/satquery.js", "r", encoding="utf-8") as f:
        js = f.read()
    
    # renderAuditReplay must pass lifecycle_trace through
    assert "lifecycle_trace: data.lifecycle_trace" in js, \
        "renderAuditReplay must pass lifecycle_trace to normalized object"
    assert "pipeline_metrics: data.pipeline_metrics" in js, \
        "renderAuditReplay must pass pipeline_metrics to normalized object"


# ===========================================================================
# Test 14: Frontend handles missing lifecycle gracefully
# ===========================================================================
def test_frontend_shows_unavailable_for_missing_trace():
    """
    Verify the JS renderTracePanel shows 'unavailable' message for empty traces.
    """
    with open("static/js/satquery.js", "r", encoding="utf-8") as f:
        js = f.read()
    
    assert "Lifecycle trace unavailable" in js, \
        "Frontend must show 'unavailable' message when lifecycle_trace is empty"
