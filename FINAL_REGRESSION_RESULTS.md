# Final Regression Results

## Regression Execution Summary

| Suite | Command | Result |
|-------|---------|--------|
| Real Hero E2E | `python -m pytest tests/integration/test_real_hero_e2e.py -q` | **1 passed** (completed successfully) |
| Golden Output | `python -m pytest tests/golden/ -q` | **9 passed** (completed successfully) |
| CRS Safety | `python -m pytest tests/unit/test_crs_safe_gis.py -q` | **7 passed** (completed successfully) |
| Targeted Regression | `python -m pytest tests/unit/ -q` | **334 passed, 24 skipped** (completed successfully) |

### Notes
* **Real Hero E2E**: Fully completes the exact T1 (Water) -> T2 (Change) -> T3 (Building) -> T4 (Metric Buffer) -> T5 (Intersection) plan using real files (`S2_T1.tif`, `S2_T2.tif`, `S1_GRD.tif`) without bypassing the orchestrator.
* **Golden Tests**: The output behaviors exactly match the strict JSON structural expectations across all verified functionalities.
* **Targeted Regression**: The core P1/P2 capabilities covering multimodality, input binding, visual traces, temporal logic, and evidence alignment strictly pass. No tests hang or block during local validation.
