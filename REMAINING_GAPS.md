# Remaining Gaps
## SATQuery AI — SIH 2026 PS26167

This document lists the remaining P1 and P2 implementation gaps based on the `CURRENT_IMPLEMENTATION_AUDIT.md`. P0 gaps have been fully addressed or honestly documented as architectural boundaries.

---

## P1 Gaps (Priority)

**All P1 Gaps are COMPLETE as of this session.**

*   **Quantitative Bi-Temporal Water Change**: COMPLETE
*   **Frontend Execution vs Verification Conflation**: COMPLETE
*   **Frontend Client-Side Confidence Threshold**: COMPLETE
*   **Replay Lifecycle Trace Audit**: COMPLETE
*   **Input Binding / Asset Router**: COMPLETE

### 4. Leaflet Map Resize Bug
**Status**: STILL MISSING
**Description**: Unhiding the map panel leaves Leaflet tiles grey until the user resizes the window.
**Required Fix**: Call `map.invalidateSize()` with a small timeout when the map panel becomes visible.

---

## P2 Gaps (Nice to Have)

### 5. Golden Regression Suite
**Status**: MISSING
**Description**: Lack of locked deterministic inputs $\rightarrow$ expected deterministic JSON outputs for the core spectral/GIS components.
**Required Fix**: Create a `tests/golden/` directory and add regression checks.

### 6. Dynamic Visual Overlays
**Status**: MISSING
**Description**: The frontend relies on hardcoded overlay image filenames for the Leaflet maps.
**Required Fix**: Dynamically link generated geometries/rasters to the map view.
