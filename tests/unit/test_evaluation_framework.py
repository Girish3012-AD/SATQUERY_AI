"""Focused unit tests for the SATQuery evaluation framework."""
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.schemas.evidence import Evidence
from src.evaluation.metrics import (
    calculate_evidence_completeness,
    calculate_provenance_completeness,
    calculate_trace_completeness,
    evaluate_routing,
    evaluate_verifier_robustness,
    collect_reproducibility_metadata,
)
from src.evaluation.benchmark_runner import SATQueryBenchmarkRunner, ROUTING_BENCHMARK


def test_reproducibility_metadata_collection():
    meta = collect_reproducibility_metadata()
    assert "timestamp" in meta
    assert "git_commit" in meta
    assert "python_version" in meta
    assert "package_versions" in meta
    assert "aoi_rasuwa" in meta
    assert meta["crs"] == "EPSG:32645"


def test_evidence_completeness_metric():
    ev = Evidence(
        evidence_id="EVAL_TEST_01",
        source="WaterSpecialist",
        task="water_detection",
        model="NDWI",
        sensor="Sentinel-2",
        modality="optical",
        timestamp="2024-10-13",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        measurement={"area": 100},
        result={},
        confidence=0.7,
        provenance={"confidence_calibration": "NOT_CALIBRATED"},
    )
    res = calculate_evidence_completeness(ev)
    assert res["evidence_schema_completeness"] == 1.0
    assert len(res["missing_fields"]) == 0


def test_provenance_completeness_metric():
    ev = Evidence(
        evidence_id="EVAL_TEST_02",
        source="WaterSpecialist",
        task="water_detection",
        model="NDWI",
        modality="optical",
        confidence=0.7,
        provenance={
            "scene_id": "scene_1",
            "sensor": "Sentinel-2",
            "formula": "NDWI",
            "threshold": 0.0,
            "confidence_calibration": "NOT_CALIBRATED",
        },
        result={},
    )
    res = calculate_provenance_completeness(ev)
    assert res["provenance_completeness"] == 1.0


def test_trace_completeness_metric():
    trace = [
        {"step": "query_received"},
        {"step": "task_controller_routing"},
        {"step": "pipeline_complete"},
    ]
    expected = ["query_received", "task_controller_routing", "pipeline_complete"]
    res = calculate_trace_completeness(trace, expected)
    assert res["trace_completeness"] == 1.0

    missing_trace = [{"step": "query_received"}]
    res2 = calculate_trace_completeness(missing_trace, expected)
    assert res2["trace_completeness"] == pytest.approx(1 / 3, abs=1e-3)
    assert "pipeline_complete" in res2["missing_steps"]


def test_routing_evaluation_benchmark():
    res = evaluate_routing(ROUTING_BENCHMARK)
    assert res["query_count"] == len(ROUTING_BENCHMARK)
    assert res["routing_success_rate"] == 1.0
    assert res["correct_routing_count"] == len(ROUTING_BENCHMARK)


def test_verifier_robustness_evaluation():
    res = evaluate_verifier_robustness()
    assert res["valid_case_accepted"] is True
    assert res["verification_valid_case_acceptance_rate"] == 1.0
    assert res["verification_invalid_case_rejection_rate"] == 1.0
    assert res["negative_cases_rejected"] == res["negative_cases_tested"]


def test_benchmark_runner_all_evaluations():
    runner = SATQueryBenchmarkRunner(output_dir="outputs/evaluation")
    report = runner.run_all_evaluations()

    assert "overall_summary" in report
    assert "reproducibility" in report
    assert "routing_evaluation" in report
    assert "verifier_evaluation" in report
    assert "workflows" in report

    summary = report["overall_summary"]
    assert summary["workflow_execution_success_rate"] == 1.0
    assert summary["routing_success_rate"] == 1.0
    assert summary["verification_valid_case_acceptance_rate"] == 1.0
    assert summary["verification_invalid_case_rejection_rate"] == 1.0


def test_multimodal_missing_modality_rejection():
    runner = SATQueryBenchmarkRunner(output_dir="outputs/evaluation")
    mm_eval = runner.evaluate_multimodal_workflow()

    assert mm_eval["optical_only_rejected_as_expected"] is True
    assert mm_eval["sar_only_rejected_as_expected"] is True
    assert mm_eval["workflow_execution_success"] is True


def test_hero_workflow_evaluation():
    runner = SATQueryBenchmarkRunner(output_dir="outputs/evaluation")
    hero_eval = runner.evaluate_hero_workflow()

    assert hero_eval["workflow_execution_success"] is True
    assert hero_eval["evidence_graph_nodes"] >= 4
    assert hero_eval["qualifying_buildings_count"] == 14
    assert hero_eval["buffer_distance_m"] == 500.0
