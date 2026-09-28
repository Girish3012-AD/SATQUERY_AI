# Input Binding / Asset Router Validation

## 1. Previous Global-Broadcast Architecture
Previously, the `ExecutionEngine` treated the `inputs` pool as a homogeneous `list[str]` and broadcast the exact same list to *every* executed Specialist. This caused heterogeneous multi-modal queries (like the Hero query needing Sentinel-1 SAR and Sentinel-2 Bi-Temporal inputs) to crash because specialists received assets they didn't require or understand.

## 2. New Classification Architecture
The existing `InputMetadataResolver` was augmented with a `classify_assets()` method. Instead of just producing Sentinel-1 JSON routing parameters, it now maps the incoming list of strings into a typed `AssetProfile` tracking `modality`, `temporal_tag`, `bands`, and `sensor`. It respects JSON sidecar files first and deterministically parses filenames as a documented fallback.

## 3. Binding Algorithm
In `SATQueryOrchestrator._bind_inputs_to_plan()`, the planner maps the classified global pool directly to specific execution `PlanStep` elements prior to execution.
The Execution Engine (`execute`) intercepts this `InputBindingMap` and provisions each step's `inputs` with only the exact file paths required, returning a deterministic `BLOCKED` status instead of crashing if required assets are missing.

## 4. Specialist Requirements
Rather than forcing all Specialists to change their `infer(inputs: list[str])` interface, a minimal declarative configuration dictionary `REQUIRED_INPUT_PROFILE` was appended to `Specialist` classes:
*   `BiTemporalWaterChangeSpecialist`: `{"modality": "optical", "temporal": "bi-temporal", "count": 2}`
*   `ChangeSpecialist`: `{"modality": "optical", "temporal": "bi-temporal", "count": 2}`
*   `BuildingDetectionSpecialist`: `{"modality": "optical"}`
*   `FloodSpecialist`: `{"modality": "optical"}`
*   `SARSpecialist`: `{"modality": "sar"}`

## 5. Temporal Ordering
A strict temporal ordering check was added for `temporal="bi-temporal"` specifications. The Router explicitly ensures T1 assets map to `inputs[0]` and T2 assets map to `inputs[1]`, retaining correct differencing behavior for existing models without rewriting them.

## 6. Missing-Input Behavior
If an asset is missing (e.g., requires bi-temporal but only T1 is present), the step binds to a failure dictionary rather than guessing. The `ExecutionEngine` translates this immediately into a `success=False` result with `BLOCKED: Missing required assets`, short-circuiting the pipeline gracefully.

## 7. Lifecycle Trace
The `INPUT_SCENE_RESOLUTION` execution stage trace in `src/orchestration/orchestrator.py` was updated to log both `classified_assets` and `input_bindings` to the artifact JSON. This guarantees that asset mapping choices remain auditable on replay without exposing binary payloads.

## 8. Backwards Compatibility
Specialists lacking a `REQUIRED_INPUT_PROFILE` receive the entire un-filtered list by default. The `ExecutionEngine` gracefully defaults to standard `inputs` parameter passing when `input_bindings` is empty, retaining full E2E test-suite pass rates without refactoring every fallback script.

## 9. Hero E2E Result
A synthetic test script (`test_hero_orchestration.py`) utilizing the `SATQueryOrchestrator` API successfully planned the DAG and proved that `BuildingUNet`, `FloodSpecialist`, and `ChangeSpecialist` all triggered successfully, isolating their necessary inputs from the provided heterogeneous global pool (`["S2_T1.tif", "S2_T2.tif", "S1_GRD.tif"]`).

## 10. Exact Tests
14 focused unit tests created in `tests/unit/test_input_binding.py`. Total regressions: 97 passed. Benchmark: 5/5.

## 11. Remaining Limitations
The system still depends partly on filename parsing when proper `.json` metadata sidecars are omitted. Wait state around 22% of the total test suite (model download blocking) remains present but unaffected by this routing upgrade.
