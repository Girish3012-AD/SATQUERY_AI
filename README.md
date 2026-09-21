# SATQuery AI — Intelligent Multimodal Satellite Imagery Geographic Reasoning

**Smart India Hackathon 2026** — Problem Statement PS26167  
**Official Repository**: `https://github.com/Girish3012-AD/SATQUERY_AI.git`  
**Current Main Commit**: `04d8949`  
**Core Architecture State**: FROZEN & CANONICAL  

---

## 🛰️ 1. Problem Statement & Executive Summary

Remote sensing analysis traditionally requires specialized GIS knowledge, manual spectral band calculations, and separate machine learning toolchains. **SATQuery AI** bridges natural language query understanding with deterministic geospatial computation and formal claim verification.

Instead of relying on end-to-end black-box LLM hallucinations for geographic facts, SATQuery enforces a three-stage paradigm:
$$\text{AI UNDERSTANDS} \longrightarrow \text{GIS COMPUTES} \longrightarrow \text{GeoReason VERIFIES}$$

- **AI UNDERSTANDS**: Deconstructs complex natural language queries into task specifications, capabilities, and spatial-temporal operations via `TaskController` and `EvidencePlanner`.
- **GIS COMPUTES**: Executes exact spectral, raster, vector, and SAR processing algorithms using specialized domain engines.
- **GeoReason VERIFIES**: Formally validates emitted evidence objects against rule-based schema contracts (`GeoReasonVerifier`) before returning answers to the user.

---

## 🏛️ 2. Core Architecture

The production execution path is strictly canonical. All frontend user actions pass through the FastAPI REST layer:

```
[ User Query / File Upload ]
            │
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Browser SPA Frontend (static/index.html + static/js/satquery.js)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST API
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ FastAPI Web Server (api_server.py)                                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ SATQueryOrchestrator (src/orchestration/orchestrator.py)                │
│   ├── TaskController (src/controller/task_controller.py)               │
│   ├── EvidencePlanner (src/planner/evidence_planner.py)               │
│   ├── ExecutionEngine & Domain Specialists                             │
│   │     ├── VqaSpecialist (Qwen2-VL-2B-Instruct + LoRA)                │
│   │     ├── WaterSpecialist (NDWI Spectral Ratioing)                   │
│   │     ├── BuildingDetectionSpecialist (ResNet/UNet Model)             │
│   │     ├── TemporalChangeSpecialist (Bialgebraic Rationing)           │
│   │     ├── SARSpecialist (Sentinel-1 Backscatter Engine)             │
│   │     └── MultimodalSpecialist (Optical + SAR Fusion)               │
│   ├── EvidenceRegistry (src/evidence/registry.py)                       │
│   └── GeoReasonVerifier (src/verifier/verifier.py)                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ JSON Response + GeoJSON Layers
                                    ▼
[ Leaflet Interactive Map + 10 Analytical UI Panels + R13 Execution Report ]
```

---

## 🗺️ 3. Supported Workflows & Query Types

1. **Single-Image Vision-Language VQA**:
   - *Example Query*: `"Describe the land-cover and major objects visible in this image."`
   - *Specialist*: `VqaSpecialist` using `Qwen2-VL-2B-Instruct` base model + LoRA adapter (`qwen2vl_rs_vqa_evidence_grounded_dev`).
2. **Single-Image Spatial Grounding**:
   - *Example Query*: `"Highlight the water body referred to in the image."`
   - *Specialist*: `WaterSpecialist` calculating NDWI $(B03 - B08) / (B03 + B08)$ and vectorizing water candidate polygons.
3. **Bi-Temporal Change Analysis**:
   - *Example Query*: `"Show spectral changes between the 2023 and 2024 Sentinel-2 observations."`
   - *Specialist*: `TemporalChangeSpecialist` performing bi-temporal spectral rationing.
4. **Multimodal Optical + SAR Reasoning**:
   - *Example Query*: `"Analyze the area using both optical and SAR evidence."`
   - *Specialists*: `MultimodalSpecialist` + `SARSpecialist` integrating Sentinel-2 MSI with Sentinel-1 C-SAR backscatter.
5. **Hero Multi-Step Geographic Reasoning**:
   - *Example Query*: `"Find newly constructed buildings within 500 m of flooded areas."`
   - *Pipeline*: Deconstructs query $\rightarrow$ Water Candidate Detection $\rightarrow$ Building Candidate Detection $\rightarrow$ Temporal Change Analysis $\rightarrow$ 500m GIS Spatial Buffer $\rightarrow$ Polygon Intersection $\rightarrow$ Formal Claim Verification.

### ⚡ One-Click Self-Contained SIH Demo Buttons
For SIH 2026 judging convenience, 5 preset demo buttons are built directly into the UI header:
- `▶ Run VQA Demo`
- `▶ Run Water Grounding Demo`
- `▶ Run Change Detection Demo`
- `▶ Run Optical + SAR Demo`
- `★ Run Hero Geographic Reasoning`

Clicking any preset automatically resolves the official validated Sentinel satellite rasters (`data/samples/vqa_test.png`, `data/samples/test.tif`) on the backend, executing the full live pipeline through `SATQueryOrchestrator` without requiring manual file uploads by judges. Manual file uploads remain fully active for custom queries.

---

## 📡 4. Real Satellite Data & AI/ML Models

### Remote Sensing Datasets
- **Sentinel-2 L2A Optical Imagery**: Multi-spectral imagery (B02 Blue, B03 Green, B04 Red, B08 NIR) covering the Rasuwa, Nepal AOI (`EPSG:32645`).
- **Sentinel-1 GRD SAR Imagery**: C-band Synthetic Aperture Radar (VV/VH polarization).

### AI/ML Models & Checkpoints
- **Base VQA Model**: `Qwen/Qwen2-VL-2B-Instruct` (Hugging Face cached snapshot).
- **LoRA RS-VQA Adapter**: `outputs/checkpoints/qwen2vl_rs_vqa_evidence_grounded_dev`.
- **Building Detection Model**: `outputs/checkpoints/building_unet_10epoch_dev.pt`.

---

## 🔬 5. Evidence & Verification System

Every analytical step generates an explicit, schema-compliant `Evidence` object containing:
- **`evidence_id`**: Deterministic unique ID (e.g. `E-VQA-001`, `E-WATER-001`, `E-CHANGE-001`).
- **`task` & `model`**: Exact task type and model/specialist used.
- **`sensor` & `modality`**: `MSI`/`C-SAR`, `optical`/`sar`.
- **`geometry`**: GeoJSON WGS84 Polygon/MultiPolygon (`EPSG:4326`).
- **`confidence`**: Raw, uncalibrated score float `[0.0, 1.0]`.
- **`provenance`**: Complete record of input parameters, band indices, and timestamp metadata.

The **`GeoReasonVerifier`** validates evidence prior to answer release:
- Asserts geometry validity, coordinate system compliance, and required timestamp existence.
- Detects contradictions between specialists.
- If verification rules pass $\rightarrow$ status `verified`. If metadata/geometry missing or invalid $\rightarrow$ status `abstain`.

---

## 🔄 6. LIVE vs REPLAY Execution Modes

- **LIVE Mode (`⚡ LIVE EXECUTION`)**: Triggered when a user submits a query or uploads an image via `POST /api/query`. The orchestrator executes the real pipeline. If inputs or checkpoints are missing, it outputs an explicit error message (never silently falling back to replay).
- **REPLAY Mode (`📂 AUDIT REPLAY`)**: Clearly labeled secondary mode for offline judge demonstration. Instantly replays verified JSON audit artifacts from `outputs/`.
- **UI Delineation**: Prominent top-right mode indicator badge clearly states `LIVE` (blue) or `REPLAY` (purple) at all times.

---

## 📁 7. File Upload & Security Hardening

The `/api/upload` endpoint enforces secure file handling:
- **Supported Formats**: `.tif`, `.tiff` (GeoTIFF), `.png`, `.jpg`, `.jpeg`.
- **Max File Size**: 50 MB stream limit.
- **Validation**:
  1. Filename sanitization via regex + prepended `UUID`.
  2. Streamed size enforcement.
  3. Binary content verification via `PIL.Image.verify()` and `rasterio.open()`.
  4. Isolation into a dedicated `uploads/` directory.

---

## ⚙️ 8. Local Setup & Execution Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.13.7)
- `pip` and PowerShell/Bash terminal

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/Girish3012-AD/SATQUERY_AI.git
cd SATQUERY_AI

# Install dependencies
pip install -r requirements.txt
```

### Launch the Web Server & SPA
```bash
# Start the FastAPI server
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```
Open your browser at **`http://localhost:8000`** to access the judge-ready frontend UI.

### Run Unit Tests
```bash
python -m pytest tests/unit/ -v --tb=short
```

### Run Reproducible Evaluation Benchmark
```bash
python run_evaluation.py
```
Outputs evaluation reports to `outputs/evaluation/evaluation_report.json` and `outputs/evaluation/EVALUATION_REPORT.md`.

---

## ⚖️ 9. Scientific Claim Boundaries & Limitations

In accordance with rigorous scientific standards:
1. **NDWI Water Detection**: NDWI $\ge 0.0$ detects **potential water candidates**, NOT confirmed flooding.
2. **Spectral Change**: Bi-temporal spectral difference indicates **spectral change candidates**, NOT confirmed building construction without high-resolution optical/LIDAR verification.
3. **Spatial Proximity**: A 500 m buffer establishes **spatial association**, NOT causal flood damage.
4. **Model Confidence**: Confidence values are **raw model/rule outputs**, NOT calibrated posterior probabilities.
5. **Multimodal Fusion**: Current fusion uses deterministic optical/SAR evidence overlay, NOT a joint learned multimodal foundation transformer.
6. **Dataset Tests**: Offline dataset training tests require raw SpaceNet4 files and are cleanly marked as `SKIPPED (DATASET_REQUIRED)` when dataset files are absent, avoiding false passes.

---

## 📊 10. Key Evaluation & Artifact References

- **Final System Validation Report**: `outputs/final_validation/FINAL_VALIDATION_REPORT.md`
- **Evaluation Benchmark Report**: `outputs/evaluation/EVALUATION_REPORT.md`
- **Hero Reasoning Map Overlay**: `outputs/hero_reasoning_overlay.png`
- **Water Grounding Map Overlay**: `outputs/grounding_water_overlay.png`
- **Judge Demo Runbook**: `DEMO_RUNBOOK.md`
- **Recommended Demo Queries**: `SIH_DEMO_QUERIES.md`
