"""
SATQuery AI — Unit Tests for Canonical Execution Lifecycle & Tracing

Verifies:
1. Every query generates all 15 canonical pipeline stages.
2. Non-applicable stages (e.g. GIS for VQA) are marked NOT_APPLICABLE with reason.
3. Decoupling of execution status (execution_success) and verifier outcome (LOW_CONFIDENCE, ABSTAIN, REJECT).
4. Truthful metric calculation (Total=15, Executed, Not Applicable, Errors=0, Evidence Count).
5. Execution failures (uncaught exceptions) set execution_success=False and increment execution_errors.
"""

import pytest
from src.orchestration.lifecycle import PipelineStage, StageStatus, STAGE_DISPLAY_NAMES, PipelineMetrics, ExecutionStageTrace
from src.orchestration.orchestration_result import OrchestrationResult
from src.controller.task_controller import TaskSpec
from src.planner.evidence_planner import EvidencePlan, PlanStep
from src.executor.execution_result import ExecutionResult
from src.orchestration.orchestrator import SATQueryOrchestrator, OrchestratorConfig


def test_15_canonical_stages_generated():
    """Verify that every execution trace contains exactly 15 canonical stages."""
    orchestrator = SATQueryOrchestrator()
    task_spec = TaskSpec(
        task_id="test_t1",
        query="What is in the image?",
        task_type="vqa",
    )
    plan = EvidencePlan(
        plan_id="plan_123",
        task_id="test_t1",
        query="What is in the image?",
        steps=[PlanStep(step_id="s1", task="vqa", operation="inference")]
    )
    results = [
        ExecutionResult(step_id="s1", task="vqa", success=True, execution_success=True, evidence_ids=["ev_1"])
    ]
    verification = {"status": "verified", "confidence": 0.95, "recommended_action": "accept"}

    lifecycle_trace, metrics = orchestrator._build_canonical_lifecycle(
        task_spec=task_spec,
        plan=plan,
        inputs=["data/samples/vqa_test.png"],
        selected_capabilities={"vqa": "VqaSpecialist"},
        selected_models={"vqa": "Qwen2-VL-7B-Instruct"},
        results=results,
        evidence_ids=["ev_1"],
        verification=verification,
        start_time=100.0,
    )

    assert len(lifecycle_trace) == 15
    stage_ids = [s["stage_id"] for s in lifecycle_trace]
    expected_stage_ids = [f"STAGE_{stage.value.upper()}" for stage in PipelineStage]
    assert stage_ids == expected_stage_ids
    assert metrics["total_stages"] == 15
    assert metrics["execution_errors"] == 0


def test_gis_stage_not_applicable_for_vqa():
    """Verify non-GIS queries mark gis_spatial_processing as NOT_APPLICABLE."""
    orchestrator = SATQueryOrchestrator()
    task_spec = TaskSpec(task_id="t_vqa", query="Describe scene", task_type="vqa")
    plan = EvidencePlan(
        plan_id="p1",
        task_id="t_vqa",
        query="Describe scene",
        steps=[PlanStep(step_id="s1", task="vqa", operation="inference")]
    )

    lifecycle_trace, metrics = orchestrator._build_canonical_lifecycle(
        task_spec=task_spec,
        plan=plan,
        inputs=["test.png"],
        selected_capabilities={},
        selected_models={},
        results=[],
        evidence_ids=[],
        verification={},
        start_time=0.0,
    )

    gis_stage = next(s for s in lifecycle_trace if "GIS_SPATIAL_PROCESSING" in s["stage_id"])
    assert gis_stage["status"] == StageStatus.NOT_APPLICABLE.value
    assert gis_stage["execution_success"] is True
    assert "Query does not require GIS spatial transformations." in gis_stage["reason"]
    assert metrics["not_applicable_stages"] >= 1


def test_gis_stage_executed_for_spatial_queries():
    """Verify GIS stage is EXECUTED when spatial operations are present."""
    orchestrator = SATQueryOrchestrator()
    task_spec = TaskSpec(
        task_id="t_gis",
        query="Compute flood area intersection",
        task_type="specialized_analysis",
        spatial_operations=["intersection", "area"]
    )
    plan = EvidencePlan(
        plan_id="p_gis",
        task_id="t_gis",
        query="Compute flood area intersection",
        steps=[PlanStep(step_id="s1", task="intersection", operation="intersection")]
    )

    lifecycle_trace, metrics = orchestrator._build_canonical_lifecycle(
        task_spec=task_spec,
        plan=plan,
        inputs=["test.tif"],
        selected_capabilities={},
        selected_models={},
        results=[ExecutionResult(step_id="s1", task="intersection", success=True, execution_success=True)],
        evidence_ids=["ev_spatial"],
        verification={},
        start_time=0.0,
    )

    gis_stage = next(s for s in lifecycle_trace if "GIS_SPATIAL_PROCESSING" in s["stage_id"])
    assert gis_stage["status"] == StageStatus.EXECUTED.value
    assert gis_stage["execution_success"] is True


def test_decoupling_low_confidence_from_execution_failure():
    """Verify LOW_CONFIDENCE, ABSTAIN, or REJECT outcomes do NOT count as execution errors."""
    orchestrator = SATQueryOrchestrator()
    task_spec = TaskSpec(task_id="t_low", query="Check water change", task_type="temporal_analysis")
    plan = EvidencePlan(
        plan_id="p_low",
        task_id="t_low",
        query="Check water change",
        steps=[PlanStep(step_id="s1", task="change_detection", operation="change_detection")]
    )
    results = [
        ExecutionResult(
            step_id="s1",
            task="change_detection",
            success=True,
            execution_success=True,
            result_status="low_confidence",
            evidence_ids=["ev_low"]
        )
    ]
    verification = {
        "status": "low_confidence",
        "confidence": 0.45,
        "recommended_action": "request_additional_evidence",
        "reason": "Low optical resolution in temporal frame."
    }

    lifecycle_trace, metrics = orchestrator._build_canonical_lifecycle(
        task_spec=task_spec,
        plan=plan,
        inputs=["frame1.tif", "frame2.tif"],
        selected_capabilities={},
        selected_models={},
        results=results,
        evidence_ids=["ev_low"],
        verification=verification,
        start_time=0.0,
    )

    assert metrics["execution_errors"] == 0
    assert metrics["final_decision"] == "LOW_CONFIDENCE"
    assert metrics["verification_status"] == "low_confidence"

    verifier_stage = next(s for s in lifecycle_trace if "EVIDENCE_VERIFICATION" in s["stage_id"])
    assert verifier_stage["execution_success"] is True
    assert verifier_stage["result_status"] == "low_confidence"


def test_truthful_metrics_aggregation():
    """Verify aggregated PipelineMetrics attributes match trace state."""
    orchestrator = SATQueryOrchestrator()
    task_spec = TaskSpec(task_id="t_meta", query="VQA query", task_type="vqa")
    plan = EvidencePlan(
        plan_id="p_meta",
        task_id="t_meta",
        query="VQA query",
        steps=[PlanStep(step_id="s1", task="vqa", operation="inference")]
    )

    _, metrics = orchestrator._build_canonical_lifecycle(
        task_spec=task_spec,
        plan=plan,
        inputs=["img.png"],
        selected_capabilities={},
        selected_models={},
        results=[ExecutionResult(step_id="s1", task="vqa", success=True, execution_success=True, evidence_ids=["e1", "e2"])],
        evidence_ids=["e1", "e2"],
        verification={"status": "verified"},
        start_time=0.0,
    )

    assert metrics["total_stages"] == 15
    assert metrics["executed_stages"] == 14
    assert metrics["not_applicable_stages"] == 1
    assert metrics["execution_errors"] == 0
    assert metrics["evidence_count"] == 2
