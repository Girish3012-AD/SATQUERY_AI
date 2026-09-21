"""
report_generator.py — Generator for evaluation_report.json and EVALUATION_REPORT.md.

Produces machine-readable JSON and human-readable GitHub Flavored Markdown reports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_evaluation_reports(report_data: dict[str, Any], output_dir: Path | str = "outputs/evaluation") -> tuple[Path, Path]:
    """Write evaluation_report.json and EVALUATION_REPORT.md."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "evaluation_report.json"
    md_path = out_dir / "EVALUATION_REPORT.md"

    # 1. Write machine-readable JSON
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    # 2. Write human-readable Markdown
    summary = report_data.get("overall_summary", {})
    repro = report_data.get("reproducibility", {})
    routing = report_data.get("routing_evaluation", {})
    verifier = report_data.get("verifier_evaluation", {})
    workflows = report_data.get("workflows", {})

    md_content = f"""# SATQuery Reproducible Evaluation Report

> **Evaluation Date**: {report_data.get("timestamp")}  
> **Git Commit**: `{repro.get("git_commit")}`  
> **Python Version**: `{repro.get("python_version")}`  
> **Platform**: `{repro.get("platform")}`  

---

## 1. Executive Summary

| Metric | Measured Value | Classification |
| :--- | :--- | :--- |
| **Workflow Execution Success Rate** | **{summary.get("workflow_execution_success_rate") * 100:.1f}%** ({summary.get("successful_workflows")}/{summary.get("total_workflows_evaluated")}) | Measured / Operational |
| **Query Routing Success Rate** | **{summary.get("routing_success_rate") * 100:.1f}%** ({routing.get("correct_routing_count")}/{routing.get("query_count")}) | Operational (Rule-based regression) |
| **Verifier Valid Case Acceptance Rate** | **{summary.get("verification_valid_case_acceptance_rate") * 100:.1f}%** | Measured |
| **Verifier Invalid Case Rejection Rate** | **{summary.get("verification_invalid_case_rejection_rate") * 100:.1f}%** | Measured |
| **Total Evaluation Runtime** | **{summary.get("total_evaluation_runtime_seconds")} s** | Operational Timing |

---

## 2. Environment & Reproducibility Metadata

- **AOI (Rasuwa, Nepal)**: `{repro.get("aoi_rasuwa")}`
- **Native CRS**: `{repro.get("crs")}`
- **Models & Checkpoints**:
  - VQA Base Model: `{repro.get("models", {}).get("vqa_base")}`
  - VQA LoRA Adapter: `{repro.get("models", {}).get("vqa_adapter")}`
  - Building Model: `{repro.get("models", {}).get("building_unet")}`
  - Water/Flood Model: `{repro.get("models", {}).get("water_ndwi")}`
  - Change Detector: `{repro.get("models", {}).get("change_detector")}`
- **Key Package Versions**:
  - `torch`: `{repro.get("package_versions", {}).get("torch")}`
  - `rasterio`: `{repro.get("package_versions", {}).get("rasterio")}`
  - `shapely`: `{repro.get("package_versions", {}).get("shapely")}`
  - `transformers`: `{repro.get("package_versions", {}).get("transformers")}`
  - `peft`: `{repro.get("package_versions", {}).get("peft")}`

---

## 3. Query Routing Regression Evaluation

- **Total Queries Evaluated**: {routing.get("query_count")}
- **Correct Routing Count**: {routing.get("correct_routing_count")}
- **Routing Success Rate**: **{routing.get("routing_success_rate") * 100:.1f}%**
- **Evaluation Type**: Rule-based routing regression evaluation (Not ML accuracy).

### Query Details
"""
    for res in routing.get("query_results", []):
        status = "✅ PASS" if res.get("routing_success") else "❌ FAIL"
        md_content += f"- **{status}**: *\"{res.get('query')}\"*\n"
        md_content += f"  - Expected Task: `{res.get('expected_task_type')}` | Actual: `{res.get('actual_task_type')}`\n"
        md_content += f"  - Capabilities: `{res.get('actual_capabilities')}`\n"

    md_content += f"""
---

## 4. Evaluated Capabilities & Workflows

### 4.1 Single-Image VQA
- **Status**: {"✅ PASS" if workflows.get("vqa", {}).get("workflow_execution_success") else "❌ FAIL"}
- **Query**: *"{workflows.get("vqa", {}).get("query")}"*
- **Base Model + Adapter**: `Qwen2-VL-2B-Instruct` + `qwen2vl_rs_vqa_evidence_grounded_dev`
- **Confidence Method**: `{workflows.get("vqa", {}).get("confidence_method")}`
- **Evidence Schema Completeness**: `{workflows.get("vqa", {}).get("evidence_schema_completeness")}`
- **Provenance Completeness**: `{workflows.get("vqa", {}).get("provenance_completeness")}`
- **Semantic Accuracy Metric**: *{workflows.get("vqa", {}).get("semantic_accuracy_metric")}*

### 4.2 Single-Image Spatial Grounding
- **Status**: {"✅ PASS" if workflows.get("grounding", {}).get("workflow_execution_success") else "❌ FAIL"}
- **Query**: *"{workflows.get("grounding", {}).get("query")}"*
- **Specialist**: `{workflows.get("grounding", {}).get("specialist")}`
- **Grounded Water Area**: `{workflows.get("grounding", {}).get("grounded_area_km2")} km²` ({workflows.get("grounding", {}).get("polygon_count")} polygons)
- **Bounding Box (CRS)**: `{workflows.get("grounding", {}).get("bbox_geographic")}`
- **Visual Overlay Artifact**: `{workflows.get("grounding", {}).get("visual_overlay_path")}`
- **Spatial IoU Metric**: *{workflows.get("grounding", {}).get("spatial_iou_metric")}*

### 4.3 Bi-Temporal Change Analysis
- **Status**: {"✅ PASS" if workflows.get("change", {}).get("workflow_execution_success") else "❌ FAIL"}
- **Timestamps**: T1=`{workflows.get("change", {}).get("t1_timestamp")}` vs T2=`{workflows.get("change", {}).get("t2_timestamp")}` (T1 < T2 verified)
- **Changed Area**: `{workflows.get("change", {}).get("changed_area_km2")} km²`
- **Change Accuracy Metric**: *{workflows.get("change", {}).get("change_accuracy_metric")}*

### 4.4 Optical + SAR Multimodal Fusion
- **Status**: {"✅ PASS" if workflows.get("multimodal_optical_sar", {}).get("workflow_execution_success") else "❌ FAIL"}
- **Fused Area**: `{workflows.get("multimodal_optical_sar", {}).get("fused_area_km2")} km²`
- **Missing Modality Tests**:
  - Optical ONLY rejected as expected: `{workflows.get("multimodal_optical_sar", {}).get("optical_only_rejected_as_expected")}`
  - SAR ONLY rejected as expected: `{workflows.get("multimodal_optical_sar", {}).get("sar_only_rejected_as_expected")}`

### 4.5 Hero Multi-Step Geographic Reasoning
- **Status**: {"✅ PASS" if workflows.get("hero_geographic_reasoning", {}).get("workflow_execution_success") else "❌ FAIL"}
- **Query**: *"{workflows.get("hero_geographic_reasoning", {}).get("query")}"*
- **Evidence Graph Nodes**: `{workflows.get("hero_geographic_reasoning", {}).get("evidence_graph_nodes")}`
- **Qualifying Buildings within 500m**: `{workflows.get("hero_geographic_reasoning", {}).get("qualifying_buildings_count")}` of `{workflows.get("hero_geographic_reasoning", {}).get("total_buildings_detected")}` detected
- **500 m Spatial Buffer**: Applied on Shapely geometries in `EPSG:32645`
- **Scientific Note**: *"{workflows.get("hero_geographic_reasoning", {}).get("scientific_note")}"*

---

## 5. GeoReasonVerifier Robustness & Negative Edge Cases

- **Valid Case Acceptance Rate**: **{verifier.get("verification_valid_case_acceptance_rate") * 100:.1f}%**
- **Negative Case Rejection Rate**: **{verifier.get("verification_invalid_case_rejection_rate") * 100:.1f}%** ({verifier.get("negative_cases_rejected")}/{verifier.get("negative_cases_tested")})

### Negative Test Results
"""
    for neg in verifier.get("negative_test_details", []):
        status = "✅ REJECTED AS EXPECTED" if neg.get("rejected_as_expected") else "❌ FAILED TO REJECT"
        md_content += f"- **{status}**: `{neg.get('test_case')}` -> Status: `{neg.get('status')}`\n"
        md_content += f"  - Reason: {neg.get('reasons')}\n"

    md_content += f"""
---

## 6. Metric Classifications & Limitations

1. **Explicit Scoring**: All scores are reported as raw counts or operational ratios.
2. **No Arbitrary AI Project Scores**: No ungrounded percentages (e.g. "92% accurate") are generated.
3. **Model Quality Metrics**: Ground-truth pixel/polygon annotations are unavailable for Rasuwa operational scenes; metrics are explicitly designated as **Operational Validation Only**.
4. **Network vs Computation**: Benchmark timing measures local inference, vectorization, and verification runtime only. Network COG downloads are separated.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path
