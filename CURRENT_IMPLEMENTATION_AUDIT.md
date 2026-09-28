# SATQuery AI — Current Implementation Audit
## SIH 2026 PS26167

**Audit Date**: 2026-09-27
**Commit**: `558c388` on `master`
**Auditor**: Automated deep-code inspection + 5 parallel component audits

> [!IMPORTANT]
> This audit follows the rule: **DONE = CODE EXISTS + TEST EXISTS + EXECUTION SUCCEEDS + RESULT IS VERIFIED**

---

## Component Status Matrix

| # | Component | Code Exists | Tests Exist | Tests Pass | E2E Verified | Status |
|---|-----------|:-----------:|:-----------:|:----------:|:------------:|:------:|
| 1 | Qwen2-VL LoRA Adapter (RS VQA) | YES | YES (24 skipped - dataset required) | YES | YES (228s CPU) | **DONE** (with caveats) |
| 2 | Remote Sensing Adaptation Proof | YES (audit json exists) | NO | N/A | N/A | **GAP** |
| 3 | Optical+SAR Fusion | YES | YES (4 tests) | YES | YES (via API) | **DONE** |
| 4 | SAR Specialist | YES | YES (tests exist) | YES | YES | **DONE** |
| 5 | Agentic Orchestration (Simple) | YES | YES (6+ tests) | YES | YES | **DONE** |
| 6 | Agentic Orchestration (Hero Multi-Step) | YES (scripted) | Partial (unit only) | YES | YES (manual script) | **GAP** |
| 7 | 15-Stage Lifecycle Tracing | YES | YES (5 tests) | YES | YES (live API) | **DONE** |
| 8 | GeoReasonVerifier | YES | YES (10+ tests) | YES | YES | **DONE** |
| 9 | Temporal Change Specialist | YES | YES (12+ tests) | YES | YES | **DONE** |
| 10 | Quantitative Bi-Temporal Water Change | NO | NO | N/A | N/A | **GAP** |
| 11 | Rasuwa Flood E2E (STAC) | YES | NO (E2E script only) | N/A | YES (script) | **PARTIAL** |
| 12 | Frontend Evidence UI | YES | YES (5 tests) | YES | YES | **DONE** (with gaps) |
| 13 | Confidence Calibration Labeling | Partial | NO | N/A | N/A | **GAP** |
| 14 | Execution vs Verification Decoupling (Backend) | YES | YES | YES | YES | **DONE** |
| 15 | Execution vs Verification Decoupling (Frontend) | NO | NO | N/A | N/A | **GAP** |
| 16 | Benchmark Evaluation | YES | YES | YES (5/5, 100%) | YES | **DONE** |
| 17 | Golden Regression Suite | NO | NO | N/A | N/A | **GAP** |
| 18 | Demo Queries (All 5) | YES | YES (via API) | YES | YES (0 errors) | **DONE** |

---

## P0 Findings (Critical)

### 1. Remote-Sensing Adaptation — **HONEST BUT INCOMPLETE**

**Reality**: The LoRA adapter is **NOT a genuine visual RS reasoner**. The repository's own internal audit (`experiment_audit_9_4H_8B_9.json`) conclusively proves:

- **Visual Sensitivity**: 0/24 predictions changed when real images replaced with blank/noise
- **Actual Behavior**: Text template reformatter that copies evidence numbers from prompt
- **Training Data**: 784 synthetic SpaceNet4 examples from a SINGLE Atlanta scene
- **BigEarthNet/RSVQA**: NOT used (only mentioned in requirements doc)

**Code Contradiction**:
- `orchestrator.py` line 155: `remote_sensing_adapted: True` ← **FALSE**
- `experiment_audit_9_4H_8B_9.json`: `remote_sensing_adapted: false` ← **TRUE**

**Architecture**:
- Base: Qwen2-VL-2B-Instruct
- LoRA: r=8, alpha=16, targets: q_proj, v_proj (language decoder only)
- Visual encoder: **FROZEN** (zero visual gradient pressure)
- Trainable params: 1,089,536 (0.049%)

**Gap**: Fix the `remote_sensing_adapted` claim in orchestrator. Create `REMOTE_SENSING_ADAPTATION_REPORT.md` documenting the honest status.

---

### 2. Optical+SAR Fusion — **LEVEL C (DONE)**

**Classification**: Level C — Late Fusion / Decision-Level Geometric Intersection

**How it works**:
1. Optical branch: NDWI >= 0.0 → water candidate polygons
2. SAR branch: VV backscatter < 0.0316 (≈-15 dB) → low-backscatter polygons
3. Fusion: Shapely `polygon.intersection(sar_union)` → joint candidates
4. Confidence: 0.85 if fused area > 0, else 0.40

**Evidence**: Repository honestly documents this in `SIH_LIMITATIONS_AND_CLAIM_BOUNDARIES.md`.

**Verified**: Rasuwa multimodal E2E audit shows 2,674 optical + 12,317 SAR → 368 fused polygons (0.3868 km²).

**Note**: Temporal window relaxation (`max_temporal_separation_seconds=9999999.0`) allows ~64-day gap between optical and SAR scenes in demo.

---

### 3. Agentic Orchestration — **RULE-BASED, NOT LLM-AGENTIC**

**Architecture**:
- `TaskController`: Keyword dictionary + regex matching (NOT an LLM agent)
- `EvidencePlanner`: Deterministic if-else templates (NOT dynamic planning)
- `SensorAwareRouter`: Metadata filter over static registry (NOT cost/accuracy-aware)

**Simple queries (VQA, grounding, change, multimodal)**: Work correctly through `orchestrator.run()` — auto-decompose into linear plan → execute → verify.

**Hero multi-step query**: `"Find newly constructed buildings within 500m of flooded areas"`
- **Does NOT run through `orchestrator.run()`** — manually scripted in `hero_reasoning_e2e.py`
- Three architectural blockers prevent orchestrated execution:
  1. No per-step input dispatching (all specialists get same flat `inputs` list)
  2. Broken multi-parent DAG dependency in EvidencePlanner for GIS intersection
  3. `TaskController` maps "within" → buffer but misses "intersection"

**Gap**: Hero query needs to execute through the orchestrator, or be honestly documented as a manually-scripted demonstration.

---

### 4. SIH Mandatory Demo Queries — **ALL 5 PASS (DONE)**

| Query | Status | Exec Errors | Stages | Evidence |
|-------|--------|:-----------:|:------:|:--------:|
| A: VQA | VERIFIED | 0 | 14/15 | 1 |
| B: Water Grounding | VERIFIED | 0 | 14/15 | 1 |
| C: Bi-Temporal Change | ABSTAIN | 0 | 14/15 | 1 |
| D: Multimodal Optical+SAR | VERIFIED | 0 | 14/15 | 3 |
| E: Hero Multi-Step | ABSTAIN | 0 | 15/15 | 4 |

---

## P1 Findings

### 5. Quantitative Bi-Temporal Water Change — **GAP**

**Current state**: `ChangeSpecialist` performs generic normalized absolute raster difference across all bands. It does NOT:
- Compute NDWI for T1 and T2 separately
- Calculate water area at T1 vs T2
- Compute percentage water expansion/contraction
- Classify persistent water vs newly inundated vs receded

**What exists**: NDWI calculation lives in `src/geospatial/spectral.py` and is used by `FloodSpecialist`/`WaterSpecialist`, but these are mono-temporal only.

**Gap**: Need a `BiTemporalWaterChangeSpecialist` or extension that:
1. Computes NDWI(T1) → water mask T1 → area T1
2. Computes NDWI(T2) → water mask T2 → area T2
3. Reports ΔA = A(T2) - A(T1) and % change

---

### 6. Rasuwa Flood E2E — **PARTIAL (Real STAC, No Unit Tests)**

**What works**: `rasuwa_flood_e2e.py` uses real Sentinel-2 L2A data from Microsoft Planetary Computer STAC API. Scene `S2B_MSIL2A_20241013T044659` correctly extracted via GDAL /vsicurl/.

**What's missing**: No pytest-compatible test wrapping the E2E flow. Script requires live internet for STAC access.

---

### 7. Frontend Evidence UI — **FUNCTIONAL WITH GAPS**

**Working**:
- 10-panel UI with evidence cards, Leaflet map, 15-stage trace
- Calibration warning banner in Panel 7
- Pipeline metrics card (stages/errors/evidence count)

**Gaps**:
1. **Status conflation**: Panel 3 "Execution Status" shows verifier verdict (VERIFIED/ABSTAIN), not pipeline execution success
2. **Fake evidence VERIFIED badge**: Client-side `confidence >= 0.6` → VERIFIED, ignoring GeoReason
3. **Missing inline uncalibrated label** on percentage values
4. **Broken audit replay**: `lifecycle_trace` not passed in `renderAuditReplay()`
5. **Leaflet `invalidateSize()`** missing → grey tiles on panel unhide
6. **Hardcoded overlay filenames**: Only 2 hardcoded PNGs checked
7. **No offline Leaflet fallback**: CDN-only loading
8. **Single file upload**: Can't upload T1+T2 temporal pairs

---

### 8. Confidence Handling — **PARTIAL**

**Backend**: Correctly marks `calibrated: false` and `calibration_note` in API response.
**Frontend**: Warning banner exists in Panel 7, but individual evidence card percentages and the main meter lack `(Uncalibrated)` inline labels.

---

## P2 Findings

### 9. Benchmark Evaluation — **DONE (100%)**
- `run_evaluation.py` → 5/5 workflows, 100% success

### 10. Golden Regression Suite — **GAP**
- No pinned input→output golden test files exist
- No `tests/golden/` directory

### 11. Test Suite Health — **269 PASSED, 0 FAILED, 24 SKIPPED**
- 45 test files across unit tests
- 24 skipped tests require offline datasets (SpaceNet4, RS-VQA)
- 5 integration tests (require model checkpoints/STAC access)

---

## Required Deliverables Status

| Deliverable | Status |
|-------------|--------|
| `CURRENT_IMPLEMENTATION_AUDIT.md` | **THIS FILE** |
| `REMAINING_GAPS.md` | NOT STARTED |
| `REMOTE_SENSING_ADAPTATION_REPORT.md` | NOT STARTED |
| `OPTICAL_SAR_VALIDATION.md` | NOT STARTED |
| `AGENTIC_WORKFLOW_VALIDATION.md` | NOT STARTED |
| `TEMPORAL_WATER_CHANGE_REPORT.md` | NOT STARTED |
| `RASUWA_FLOOD_E2E_REPORT.md` | NOT STARTED |
| `FINAL_VALIDATION_REPORT.md` | NOT STARTED |
| `SIH26167_FINAL_COMPLIANCE.md` | NOT STARTED |

---

## Priority-Ordered Gap Closure Plan

### P0 (Must Fix)
1. **Fix `remote_sensing_adapted` lie** in orchestrator.py → set to `False`, create honest report
2. **Document hero query limitation** honestly OR fix orchestrator to support multi-specialist dispatch
3. **Create all P0 report deliverables**

### P1 (Should Fix)
4. **Implement bi-temporal NDWI water area comparison** (T1 vs T2 quantitative change)
5. **Decouple execution status from verifier outcome in frontend** (Panel 3)
6. **Fix evidence card VERIFIED badge** (remove client-side confidence threshold)
7. **Add inline uncalibrated labels** to confidence percentages
8. **Fix Leaflet `invalidateSize()`** bug
9. **Fix audit replay** to include lifecycle_trace

### P2 (Nice to Have)
10. **Create golden regression test suite**
11. **Dynamic visual overlays** instead of hardcoded filenames
12. **Offline Leaflet fallback**
