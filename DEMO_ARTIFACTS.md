# SATQuery AI — Presentation & Evaluation Demo Artifacts

This document enumerates all pre-generated, verified artifacts in the repository available for judge presentation, offline verification, and evaluation auditing.

---

## 📄 1. System Reports

| Artifact Path | Description | Key Metric / Information |
|---|---|---|
| [outputs/final_validation/FINAL_VALIDATION_REPORT.md](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/final_validation/FINAL_VALIDATION_REPORT.md) | Human-readable final system validation report | 269 PASSED, 0 FAILED, 24 DATASET_REQUIRED SKIPPED |
| [outputs/final_validation/final_validation_report.json](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/final_validation/final_validation_report.json) | Machine-readable validation JSON | System timestamp, canonical pipeline path, benchmark pass rate |
| [outputs/evaluation/EVALUATION_REPORT.md](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/evaluation/EVALUATION_REPORT.md) | Reproducible benchmark evaluation report | 100% workflow execution success rate & verifier acceptance rates |
| [outputs/evaluation/evaluation_report.json](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/evaluation/evaluation_report.json) | Benchmark raw metrics JSON | Detailed precision/recall/IoU metrics across specialists |

---

## 🖼️ 2. Visual Map Overlays

| Artifact Path | Description | Visual Details |
|---|---|---|
| [outputs/hero_reasoning_overlay.png](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/hero_reasoning_overlay.png) | Hero multi-step reasoning visual overlay | Rendered Sentinel-2 scene with 500m buffer zone & candidate building polygon intersections |
| [outputs/grounding_water_overlay.png](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/grounding_water_overlay.png) | Water spatial grounding visual overlay | Rendered Sentinel-2 scene with NDWI water candidate polygon outline |

---

## 📂 3. Verified Audit JSON Files (Offline Replay Engine)

These files power the **Audit Replay** secondary mode (`📂 AUDIT REPLAY`) in the frontend UI:

| Audit File Path | Size | Scene / Workflow |
|---|---|---|
| [outputs/single_image_vqa_e2e_audit.json](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/single_image_vqa_e2e_audit.json) | 12.0 KB | Qwen2-VL-2B-Instruct + LoRA VQA scene description |
| [outputs/single_image_grounding_e2e_audit.json](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/single_image_grounding_e2e_audit.json) | 58.2 KB | Water candidate spatial grounding polygon |
| [outputs/bi_temporal_change_e2e_audit.json](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/bi_temporal_change_e2e_audit.json) | 42.1 KB | Bi-temporal spectral change detection (2023 vs 2024) |
| [outputs/rasuwa_multimodal_flood_e2e_audit.json](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/rasuwa_multimodal_flood_e2e_audit.json) | 3.0 MB | Optical Sentinel-2 + SAR Sentinel-1 Rasuwa AOI multimodal fusion |
| [outputs/rasuwa_flood_e2e_audit.json](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/rasuwa_flood_e2e_audit.json) | 2.9 KB | Rasuwa optical water candidate detection |
| [outputs/rasuwa_sar_e2e_audit.json](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/rasuwa_sar_e2e_audit.json) | 6.7 KB | Sentinel-1 SAR C-band backscatter analysis |
| [outputs/hero_reasoning_e2e_audit.json](file:///d:/downloads/SATQUERY_AI-main/SATQUERY_AI-main/outputs/hero_reasoning_e2e_audit.json) | 467.4 KB | Hero multi-step geographic reasoning full execution trace |

---

## 🛰️ 4. Satellite Imagery & Checkpoint Artifacts

- **LoRA Adapter Checkpoint**: `outputs/checkpoints/qwen2vl_rs_vqa_evidence_grounded_dev`
- **Building Detection UNet**: `outputs/checkpoints/building_unet_10epoch_dev.pt`
- **Execution Traces**: `outputs/traces/trace_*.json`

---

## ⚡ 5. One-Click SIH Demo Auto-Resolved Input Mapping

When judges click any of the official preset buttons in the UI, the backend automatically resolves the following real satellite imagery paths from `DEMO_INPUT_REGISTRY` without manual file upload:

| Preset Key | Preset UI Button | Target Satellite Input | Specialist Execution Path |
|---|---|---|---|
| `vqa` | `▶ Run VQA Demo` | `data/samples/vqa_test.png` | `VqaSpecialist` (`Qwen2-VL-2B-Instruct` + LoRA) |
| `grounding` | `▶ Run Water Grounding Demo` | `data/samples/test.tif` | `WaterSpecialist` (NDWI Spectral Ratioing) |
| `change` | `▶ Run Change Detection Demo` | `data/samples/test.tif` (T1, T2) | `TemporalChangeSpecialist` (Bialgebraic Rationing) |
| `multimodal` | `▶ Run Optical + SAR Demo` | `data/samples/test.tif` (Optical, SAR) | `MultimodalFloodSpecialist` + `SARSpecialist` |
| `hero` | `★ Run Hero Geographic Reasoning` | `data/samples/test.tif` (Dual Obs) | Full Hero Pipeline (6 Evidence-Producing Steps) |
