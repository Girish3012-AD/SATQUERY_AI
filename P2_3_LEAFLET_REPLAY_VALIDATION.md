# P2-3 Leaflet and Replay Robustness Validation

## Files Changed
- `static/js/satquery.js`: Updated to handle map resilience, empty states, layered controls, replay extraction, and `map-no-geometry` fallback text.
- `tests/unit/test_p2_3_leaflet_replay.py`: Created 6 robust backend parsing and payload structuring tests covering legacy, blocked, and active replay states.

## Architecture
The SATQuery Leaflet integration maintains a robust abstraction layer over the backend's output schema. All live orchestrator artifacts (including geometries) follow standard deterministic serialization which allows the frontend `satquery.js` to blindly hydrate both Live Executions and Replay Artifacts. 

During replay, the frontend leverages the `extractGeometriesFromRaw` recursive function to restore geospatial geometries encoded within the nested artifact payload without relying strictly on the `evidence` array, maintaining backwards compatibility with legacy structures.

## Live Rendering Behavior
1. **Valid Features**: Successfully parses Points, LineStrings, Polygons, and MultiPolygons by streaming them into `L.geoJSON`.
2. **Missing Geometry Fallback**: Displays a truthful overlaid "Spatial geometry unavailable" text indicator instead of crashing, falling back to a default viewport.
3. **Empty Map Protection**: Correctly handles `bounds` errors if layers have 0 coordinates, omitting invalid bounds calculations.
4. **Dynamic Layers**: Generates grouped semantic Leaflet controls (`Water`, `Optical`, `Buildings`, `Change`, `SAR`, etc.) based on available data, dropping absent layer panels to avoid UI clutter.

## Replay Behavior
Replays parse exact matching state to live mode by wrapping the execution report in an identical structural facade (`normalized` object). 
- **Saved Evidence**: Retains original verification states and geometry polygons without recalculating model outputs.
- **Trace Persistence**: Retains accurate stage timelines.
- **Identical Behavior**: Reconstructed elements accurately trigger bidirectional Map ↔ Card linkages (Map polygons highlight cards, cards pan/flash bounds on map).

## Error Handling
1. **Legacy Artifacts**: Handled smoothly by allowing null attributes (empty `pipeline_metrics`, `lifecycle_trace`).
2. **Blocked/Failed Execution**: Frontend properly avoids attempting to render missing or malformed execution traces gracefully.
3. **Double Rendering Guard**: The `clearMap` teardown cleanly wipes previously loaded elements, layer controls, and `this.mapFeaturesByEvidenceId` dictionary linkages prior to rendering new results to avoid memory/layer leaks.

## Exact Test Results
1. **P2-2 Focused Visual Evidence Tests**: 6 passed.
2. **P2-3 Leaflet Replay Tests**: 6 passed (Valid schema structure, empty legacy states, blocked execution handling).
3. **Golden Regression**: 9 passed (No orchestration logic or payload was modified; regressions preserved).
4. **Targeted Regression**: 106 passed.

All suites are thoroughly GREEN. No geometry has been fabricated.

## Known Limitations
1. Replays depend on standard HTTP response payloads from the `/api/audits/` endpoints; if legacy artifact files are excessively un-schema'd, geometry extraction may fail to locate deep custom nesting.
2. The "Spatial geometry unavailable" indicator renders over the global map, which lacks targeted spatial framing without available bounds.
