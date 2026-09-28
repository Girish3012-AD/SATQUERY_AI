import json
import os
from pathlib import Path
from typing import Any
import pytest

from src.orchestration.orchestrator import SATQueryOrchestrator
from src.schemas.evidence import Evidence
from src.executor.specialist import Specialist

FIXTURE_DIR = Path(__file__).parent / "fixtures"

def _normalize_dict(data: Any) -> Any:
    """Recursively removes non-deterministic IDs and timestamps for golden matching."""
    if isinstance(data, dict):
        normalized = {}
        for k, v in data.items():
            # Skip highly variable or non-deterministic keys
            if k in (
                "task_id", "plan_id", "step_id", "id", "evidence_id",
                "timestamp", "time", "duration", "started_at", "completed_at",
                "evidence_ids", "depends_on"  # depends_on contains step_ids
            ):
                continue
            
            # Sometimes messages contain UUIDs or file paths
            if k == "message" and isinstance(v, str) and "BLOCKED" in v:
                normalized[k] = v  # Keep blocked reasons
                continue
            elif k in ("message", "messages"):
                continue

            # In trace, remove full step lists or outputs that contain local temp paths
            if k == "outputs" and isinstance(v, dict):
                # Clean up local file paths in outputs
                clean_outputs = {}
                for ok, ov in v.items():
                    if ok == "input_bindings":
                        # Just store basenames for bindings
                        clean_outputs[ok] = {
                            step: [Path(p).name for p in paths] if isinstance(paths, list) else paths
                            for step, paths in ov.items()
                        }
                    elif ok == "classified_assets":
                        clean_outputs[ok] = [
                            {ak: av for ak, av in a.items() if ak != "path" and ak != "metadata_source"}
                            for a in ov
                        ]
                normalized[k] = clean_outputs
                continue
            
            normalized[k] = _normalize_dict(v)
        return normalized
    elif isinstance(data, list):
        return [_normalize_dict(x) for x in data]
    return data

def assert_golden(name: str, actual_data: dict):
    normalized = _normalize_dict(actual_data)
    fixture_path = FIXTURE_DIR / f"{name}.json"
    
    if not fixture_path.exists():
        fixture_path.write_text(json.dumps(normalized, indent=2))
        pytest.skip(f"Generated missing golden fixture: {name}.json")
    
    expected = json.loads(fixture_path.read_text())
    assert normalized == expected

@pytest.fixture(autouse=True)
def mock_execution(monkeypatch):
    """Mock execution to avoid loading large models or real rasters."""
    from src.executor.executor import ExecutionEngine
    from src.executor.execution_result import ExecutionResult
    
    def fake_execute_specialist(self, step, specialist_inputs, **kwargs):
        # Allow missing inputs to fail if step was blocked
        return ExecutionResult(
            success=True,
            step_id=step.step_id,
            task=step.task,
            message="Mocked execution",
        )
    # Actually wait, execute_step handles the input_bindings blocked error, so we patch execute_specialist
    monkeypatch.setattr(ExecutionEngine, "execute_specialist", fake_execute_specialist)

def extract_structured_output(result) -> dict:
    """Extract deterministic orchestrator output for golden tests."""
    return {
        "success": result.success,
        "query": result.query,
        "task_type": result.task_type,
        "selected_capabilities": result.selected_capabilities,
        "execution_status": getattr(result, "execution_status", None),
        "verification_status": getattr(result, "verification_status", None),
        "pipeline_metrics": result.pipeline_metrics,
        "lifecycle_trace": [
            {
                "stage": str(s.get("stage_name", "")),
                "status": str(s.get("status", "").value if hasattr(s.get("status", ""), "value") else s.get("status", "")),
                "outputs": s.get("outputs", {})
            }
            for s in (result.lifecycle_trace or [])
        ]
    }

def test_single_image_vqa():
    orchestrator = SATQueryOrchestrator()
    result = orchestrator.run(
        query="What is in this image?",
        inputs=["S2_T1.tif"]
    )
    assert_golden("single_image_vqa", extract_structured_output(result))

def test_water_grounding():
    orchestrator = SATQueryOrchestrator()
    result = orchestrator.run(
        query="Where is the water?",
        inputs=["S2_T1.tif"]
    )
    assert_golden("water_grounding", extract_structured_output(result))

def test_bi_temporal_change():
    orchestrator = SATQueryOrchestrator()
    result = orchestrator.run(
        query="Show me what changed between these two images",
        inputs=["S2_T2.tif", "S2_T1.tif"]  # Reverse order to check sorting
    )
    assert_golden("bi_temporal_change", extract_structured_output(result))

def test_quantitative_bi_temporal_water_change():
    orchestrator = SATQueryOrchestrator()
    result = orchestrator.run(
        query="Calculate the water area change between T1 and T2",
        inputs=["S2_T1.tif", "S2_T2.tif"]
    )
    assert_golden("quantitative_bi_temporal_water_change", extract_structured_output(result))

def test_optical_and_sar_routing():
    orchestrator = SATQueryOrchestrator()
    # Mock capability requirement slightly or just rely on hero
    # Query that triggers multiple modalities
    result = orchestrator.run(
        query="Find floods and analyze SAR signature",
        inputs=["S2_T1.tif", "S1_GRD.tif"]
    )
    assert_golden("optical_and_sar_routing", extract_structured_output(result))

def test_hero_dag_planning_and_binding():
    orchestrator = SATQueryOrchestrator()
    result = orchestrator.run(
        query="Find newly constructed buildings within 500 m of flooded areas.",
        inputs=["S2_T1.tif", "S2_T2.tif", "S1_GRD.tif"]
    )
    assert_golden("hero_dag_planning_and_binding", extract_structured_output(result))

def test_execution_vs_verification_separation():
    orchestrator = SATQueryOrchestrator()
    result = orchestrator.run(
        query="Where is the water?",
        inputs=["S2_T1.tif"]
    )
    output = extract_structured_output(result)
    assert "execution_status" in output
    assert "verification_status" in output
    assert_golden("execution_vs_verification", output)

def test_missing_input_blocked_behavior():
    orchestrator = SATQueryOrchestrator()
    # Needs bi-temporal optical, but provide only SAR
    result = orchestrator.run(
        query="Calculate the water area change between T1 and T2",
        inputs=["S1_GRD.tif"]
    )
    
    out = extract_structured_output(result)
    assert out["success"] is False
    assert_golden("missing_input_blocked", out)

def test_lifecycle_trace_structure():
    orchestrator = SATQueryOrchestrator()
    result = orchestrator.run(
        query="Find newly constructed buildings",
        inputs=["S2_T2.tif"]
    )
    
    stages = [s.get("stage_name", "") for s in result.lifecycle_trace]
    assert "Query Received" in stages
    assert "Input Scene Resolution" in stages
    assert "Response Generation" in stages
    assert_golden("lifecycle_trace_structure", extract_structured_output(result))
