# SATQuery AI — SIH 2026 Final Compliance Audit

| Requirement | Implementation | Validation Status | Evidence File/Test | Limitations |
|-------------|----------------|-------------------|--------------------|-------------|
| **1. Single-image VQA** | Native VQA query parsing and specialist execution on GeoTIFF/PNG. | ✅ **VERIFIED** | `tests/unit/test_executor.py`, `01_single_vqa.png` | None |
| **2. Additional single-image task** | Water extraction via NDWI/spectral analysis. | ✅ **VERIFIED** | `tests/unit/test_grounding_e2e.py`, `02_water_grounding.png` | Resolution limited by native sensor. |
| **3. Bi-temporal change analysis** | Dynamic masking across strictly ordered T1/T2 pairs using `BiTemporalChangeSpecialist`. | ✅ **VERIFIED** | `tests/golden/fixtures/bi_temporal_change.json` | Requires exact T1/T2 tags. |
| **4. Optical + SAR paired analysis** | Heterogeneous inputs bind to independent models; fused into single GeoJSON. | ✅ **VERIFIED** | `tests/integration/test_real_hero_e2e.py`, `04_optical_sar.png` | Late/decision-level fusion only (No pixel-level). |
| **5. Agentic orchestration** | Planner maps natural language queries to dynamic DAGs. | ✅ **VERIFIED** | `tests/unit/test_p1_3_lifecycle_trace.py` | Requires rigid plan steps formatting. |
| **6. Specialist/model selection** | SensorAwareRouter maps task tags to specific capability specialists. | ✅ **VERIFIED** | `test_hero_geographic_workflow` | Cannot dynamically define new specialists at runtime. |
| **7. Input compatibility and asset binding** | Declarative `REQUIRED_INPUT_PROFILE` guarantees assets are properly isolated by modality. | ✅ **VERIFIED** | `tests/unit/test_input_binding.py` | Modality profiles must be explicitly defined. |
| **8. Visual/spatial evidence** | All insights generate valid WGS84 GeoJSON rendered directly in Leaflet. | ✅ **VERIFIED** | `tests/unit/test_p2_2_visual_evidence.py` | Very large polygons may stutter Leaflet. |
| **9. GIS reasoning** | Safe metric buffering dynamically computes UTM zones, safely blocking unprojectable geographic bounds. | ✅ **VERIFIED** | `tests/unit/test_crs_safe_gis.py` | Abstains on coordinates outside valid bounds. |
| **10. Verification** | Strict separation of `VerificationExecutor` analyzing topology/statistics. | ✅ **VERIFIED** | `tests/unit/test_p1_2_status_decoupling.py` | Evaluates topology and types, not ML accuracy. |
| **11. Lifecycle trace** | Continuous 15-stage bi-temporal canonical pipeline recording. | ✅ **VERIFIED** | `tests/golden/fixtures/lifecycle_trace_structure.json` | Tied heavily to specific hardcoded stage names. |
| **12. Report generation** | Automated Markdown report compilation from final `EvidencePlan`. | ✅ **VERIFIED** | `src/orchestration/orchestrator.py` | Visual report structure cannot be deeply customized. |
| **13. GeoTIFF/TIFF support** | Uses `rasterio` for deep band-level parsing and metadata extraction. | ✅ **VERIFIED** | `tests/unit/temporal_water/test_bi_temporal_water_specialist.py` | Requires intact geotransform logic. |
| **14. Remote-sensing model adaptation** | VLM evaluated against visual counterfactuals. | ⚠️ **BLOCKED** | `RS_VLM_ADAPTATION_BLOCKER.md` | Lacks local GPU resources for LoRA tuning; existing model lacks visual sensitivity. |
