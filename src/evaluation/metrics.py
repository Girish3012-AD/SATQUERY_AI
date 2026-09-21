"""
metrics.py — Scientifically honest metric calculations for SATQuery evaluation.

Calculates:
  - evidence_schema_completeness
  - provenance_completeness
  - trace_completeness
  - routing_success_rate
  - verification_valid_case_acceptance_rate
  - verification_invalid_case_rejection_rate
  - reproducibility metadata
"""
from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any

from src.controller.task_controller import TaskController
from src.schemas.evidence import Evidence
from src.verifier.georeason_verifier import GeoReasonVerifier


def calculate_evidence_completeness(evidence: Evidence) -> dict[str, Any]:
    """Calculate schema completeness for an Evidence object."""
    required_fields = [
        "evidence_id",
        "source",
        "task",
        "model",
        "modality",
        "confidence",
        "result",
        "provenance",
        "measurement",
    ]

    present = [f for f in required_fields if getattr(evidence, f, None) is not None]

    # Additional geometry check for spatial tasks
    spatial_tasks = {"flood_detection", "water_detection", "building_detection", "spatial_analysis"}
    if evidence.task in spatial_tasks:
        required_fields.append("geometry")
        if evidence.geometry is not None:
            present.append("geometry")

    score = len(present) / len(required_fields)

    return {
        "evidence_id": evidence.evidence_id,
        "task": evidence.task,
        "required_fields_count": len(required_fields),
        "present_fields_count": len(present),
        "missing_fields": [f for f in required_fields if f not in present],
        "evidence_schema_completeness": round(score, 4),
    }


def calculate_provenance_completeness(evidence: Evidence) -> dict[str, Any]:
    """Calculate provenance completeness for an Evidence object."""
    prov = evidence.provenance or {}
    base_required = ["confidence_calibration"]

    if evidence.task == "vqa":
        base_required.extend(["inference_type", "model_path", "adapter_loaded"])
    elif evidence.task in {"flood_detection", "water_detection"}:
        base_required.extend(["scene_id", "sensor", "formula", "threshold"])
    elif evidence.task == "temporal_analysis":
        base_required.extend(["t1_timestamp", "t2_timestamp", "method", "crs"])
    elif evidence.modality == "optical_sar":
        base_required.extend(["optical_evidence_id", "sar_evidence_id", "temporal_separation_seconds"])

    present = [k for k in base_required if k in prov and prov[k] is not None]
    score = len(present) / len(base_required) if base_required else 1.0

    return {
        "evidence_id": evidence.evidence_id,
        "task": evidence.task,
        "required_provenance_keys": base_required,
        "present_provenance_keys": present,
        "missing_provenance_keys": [k for k in base_required if k not in present],
        "provenance_completeness": round(score, 4),
    }


def calculate_trace_completeness(trace: list[dict[str, Any]], expected_steps: list[str]) -> dict[str, Any]:
    """Calculate completeness of execution trace steps."""
    found_steps = {entry.get("step") for entry in trace if "step" in entry}
    present = [step for step in expected_steps if step in found_steps]
    score = len(present) / len(expected_steps) if expected_steps else 1.0

    return {
        "expected_steps_count": len(expected_steps),
        "found_steps_count": len(present),
        "missing_steps": [step for step in expected_steps if step not in found_steps],
        "trace_completeness": round(score, 4),
    }


def evaluate_routing(query_benchmark: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Evaluate TaskController query routing against a rule-based query benchmark.

    Benchmark items:
      {"query": "...", "expected_task_type": "...", "expected_capabilities": [...]}
    """
    tc = TaskController()
    results = []
    correct_count = 0

    for item in query_benchmark:
        query = item["query"]
        expected_type = item["expected_task_type"]
        expected_caps = set(item.get("expected_capabilities", []))

        try:
            spec = tc.build_task_spec(query, input_count=item.get("input_count", 1))
            type_match = spec.task_type == expected_type
            caps_match = expected_caps.issubset(set(spec.required_capabilities))
            success = type_match and caps_match

            if success:
                correct_count += 1

            results.append({
                "query": query,
                "expected_task_type": expected_type,
                "actual_task_type": spec.task_type,
                "expected_capabilities": list(expected_caps),
                "actual_capabilities": spec.required_capabilities,
                "spatial_operations": spec.spatial_operations,
                "routing_success": success,
            })
        except Exception as exc:
            results.append({
                "query": query,
                "expected_task_type": expected_type,
                "routing_success": False,
                "error": str(exc),
            })

    rate = correct_count / len(query_benchmark) if query_benchmark else 0.0

    return {
        "query_count": len(query_benchmark),
        "correct_routing_count": correct_count,
        "routing_success_rate": round(rate, 4),
        "query_results": results,
    }


def evaluate_verifier_robustness() -> dict[str, Any]:
    """
    Evaluate GeoReasonVerifier against valid positive evidence and negative edge cases.
    """
    verifier = GeoReasonVerifier(minimum_confidence=0.5)

    # 1. Valid positive evidence
    valid_ev = Evidence(
        evidence_id="EVAL_VALID_001",
        source="WaterSpecialist",
        task="water_detection",
        model="NDWI_Sentinel2",
        sensor="Sentinel-2",
        modality="optical",
        timestamp="2024-10-13",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
        measurement={"water_area_km2": 0.5},
        result={"crs": "EPSG:32645"},
        confidence=0.7,
        provenance={"confidence_calibration": "NOT_CALIBRATED"},
        metadata={},
    )

    valid_res = verifier.verify([valid_ev], expected_task="water_detection")
    valid_accepted = valid_res.verified is True

    # 2. Negative test cases
    negatives = [
        ("missing_geometry", Evidence(
            evidence_id="NEG_NO_GEOM", source="WaterSpecialist", task="water_detection",
            model="NDWI", modality="optical", geometry=None, confidence=0.7, provenance={}, result={}
        ), "spatial_analysis"),
        ("missing_temporal_metadata", Evidence(
            evidence_id="NEG_NO_TEMP", source="ChangeSpecialist", task="temporal_analysis",
            model="Change", modality="optical", geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            confidence=0.7, provenance={}, result={}  # missing t1_timestamp / t2_timestamp
        ), "temporal_analysis"),
        ("low_confidence", Evidence(
            evidence_id="NEG_LOW_CONF", source="WaterSpecialist", task="water_detection",
            model="NDWI", modality="optical", geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            confidence=0.2, provenance={}, result={}  # confidence < 0.5
        ), "water_detection"),
        ("missing_multimodal_provenance", Evidence(
            evidence_id="NEG_NO_MULTI_PROV", source="MultimodalExecutor", task="multimodal_flood_detection",
            model="OpticalSAR", modality="optical_sar", geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            confidence=0.7, provenance={}, result={}  # missing optical_evidence_id / sar_evidence_id
        ), "multimodal_flood_detection"),
    ]

    rejected_count = 0
    neg_results = []

    for name, ev, exp_task in negatives:
        res = verifier.verify([ev], expected_task=exp_task)
        rejected = res.verified is False
        if rejected:
            rejected_count += 1
        neg_results.append({
            "test_case": name,
            "status": res.status,
            "rejected_as_expected": rejected,
            "reasons": res.reasons,
        })

    acceptance_rate = 1.0 if valid_accepted else 0.0
    rejection_rate = rejected_count / len(negatives) if negatives else 0.0

    return {
        "valid_case_tested": 1,
        "valid_case_accepted": valid_accepted,
        "verification_valid_case_acceptance_rate": acceptance_rate,
        "negative_cases_tested": len(negatives),
        "negative_cases_rejected": rejected_count,
        "verification_invalid_case_rejection_rate": round(rejection_rate, 4),
        "negative_test_details": neg_results,
    }


def collect_reproducibility_metadata() -> dict[str, Any]:
    """Collect system & environment metadata for evaluation reproducibility."""
    import subprocess
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        pass

    pkg_versions = {}
    for pkg in ["torch", "rasterio", "shapely", "transformers", "peft", "pillow", "pydantic"]:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkg_versions[pkg] = "not_installed"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": pkg_versions,
        "aoi_rasuwa": [85.0, 28.0, 85.5, 28.5],
        "crs": "EPSG:32645",
        "models": {
            "vqa_base": "Qwen/Qwen2-VL-2B-Instruct",
            "vqa_adapter": "outputs/checkpoints/qwen2vl_rs_vqa_evidence_grounded_dev",
            "building_unet": "outputs/checkpoints/building_unet_10epoch_dev.pt",
            "water_ndwi": "NDWI_Sentinel2_WaterGrounding",
            "sar_rtc": "Sentinel1_RTC_vv_dB_Threshold",
            "change_detector": "normalized_absolute_raster_difference",
        },
    }
