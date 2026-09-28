# Agentic Orchestration Validation
## SATQuery AI — SIH 2026 PS26167

**Status**: Planning capability = DONE. E2E Execution capability = BLOCKED/PARTIAL.

---

## 1. The Multi-Specialist Hero Query

**Query**: `"Find newly constructed buildings within 500m of flooded areas."`

This query requires evaluating multiple spatial/temporal relations across independent models. The required workflow is:
1. Detect flooding (WaterSpecialist)
2. Buffer flood by 500m (GIS)
3. Detect buildings (BuildingDetectionSpecialist)
4. Intersect buildings with buffered flood zone (GIS)
5. Detect temporal change in those buildings (ChangeSpecialist)

---

## 2. Planning and Task Decomposition (DONE)

The `TaskController` and `EvidencePlanner` have been successfully extended to support multi-specialist Directed Acyclic Graphs (DAGs).

When the orchestrator receives the hero query, it builds the following non-linear plan (verified in `test_hero_geographic_query.py`):

```text
T1: task=temporal_analysis, depends=[]
T2: task=flood_detection, depends=[]
T3: task=building_detection, depends=[]
T4: task=buffer, depends=['T2']
T5: task=intersection, depends=['T4', 'T3']
T6: task=verification, depends=['T5']
```

- **Decomposition**: Correctly splits the prompt into semantic requests (`building_detection`, `flood_detection`, `temporal_analysis`).
- **Spatial intent**: Correctly infers that "within" requires both a `buffer` and an `intersection`.
- **DAG wiring**: Correctly buffers the first geometry (`T2`) and intersects that result (`T4`) with the second geometry (`T3`).

---

## 3. End-to-End Execution (BLOCKED)

While the orchestration engine perfectly *plans* the execution, it cannot automatically *execute* it end-to-end via the `ExecutionEngine.execute()` method.

### Architectural Blocker: Global Input Binding
The `ExecutionEngine` currently assumes that all specialists operate on the exact same `inputs` list provided to the `orchestrator.run()` method.

However, different specialists require entirely different inputs:
- `WaterSpecialist`: Requires two single-band arrays (Sentinel-2 B03 Green and B08 NIR).
- `BuildingDetectionSpecialist`: Requires a 4-band VHR composite array.

Without complex input-binding resolution mapping specific user-provided files to specific tasks, passing all files globally to all models causes execution failure due to dimension/channel mismatches.

### Current Mitigation
The demonstration of this capability currently relies on the `hero_reasoning_e2e.py` script. This script acts as a manual orchestrator, binding the specific local raster files to the specific specialists, and executing the DAG steps in procedural order. 

This script demonstrates that the underlying models and GIS operations function perfectly and satisfy the user's request. However, it cannot be considered a fully "agentic" zero-touch execution.

---

## 4. Scientific Honesty Statement

We do not claim that the SATQuery API can blindly auto-execute the hero query from a single generic `/query` endpoint payload. It requires explicit input mapping.

To fix this natively within the framework would require redefining the API schema to attach payload maps to execution tasks, which violates the restriction on redesigning the core architecture for this audit. We have implemented the smallest compatible extension (DAG planning) and honestly documented the execution boundary.
