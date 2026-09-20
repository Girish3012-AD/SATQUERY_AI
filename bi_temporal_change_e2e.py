"""
bi_temporal_change_e2e.py — Genuine Bi-temporal Change E2E Pipeline.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.controller.task_controller import TaskController
from src.data.aoi_resolver import resolve_aoi, extract_location_from_query
from src.data.sentinel2_stac import Sentinel2STACDiscovery
from src.executor.change_specialist import ChangeSpecialist
from src.executor.temporal_change_specialist import TemporalChangeSpecialist
from src.evidence.registry import EvidenceRegistry
from src.verifier.georeason_verifier import GeoReasonVerifier
from src.orchestration.orchestrator import SATQueryOrchestrator, OrchestratorConfig
import rasterio

QUERY = "What changed between these two dates, and where did the change occur?"
DATETIME_RANGE_T1 = "2023-10-01/2023-10-31"  # Year before
DATETIME_RANGE_T2 = "2024-10-01/2024-10-31"
OUTPUT_DIR = Path("outputs")
AUDIT_FILE = OUTPUT_DIR / "bi_temporal_change_e2e_audit.json"

def _sign_href(href: str) -> str:
    import planetary_computer as pc
    return pc.sign(href)

def _download_aoi_window(href: str, aoi_bbox: list[float], dest: Path) -> Path:
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds

    signed = _sign_href(href)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="tif,tiff",
        VSI_CACHE=True
    ):
        with rasterio.open(signed) as src:
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, *aoi_bbox
            )
            window = from_bounds(left, bottom, right, top, transform=src.transform)
            window = window.round_lengths().round_offsets()
            data = src.read(1, window=window)
            win_transform = src.window_transform(window)
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": window.height,
                "width": window.width,
                "transform": win_transform
            })
            with rasterio.open(dest, "w", **meta) as dst:
                dst.write(data, 1)
    return dest

def run_bi_temporal_pipeline() -> dict[str, Any]:
    start_time = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    audit: dict[str, Any] = {}
    tmpdir = tempfile.mkdtemp(prefix="satquery_change_")
    
    # 1. Query & AOI
    print("\n[1] Query Understanding")
    controller = TaskController()
    task_spec = controller.build_task_spec(query=QUERY, input_count=2)
    location = extract_location_from_query(QUERY) or "Rasuwa, Nepal"
    aoi = resolve_aoi(location)
    audit["aoi"] = {"bbox": aoi.bbox}
    audit["task_spec"] = {"type": task_spec.task_type, "caps": task_spec.required_capabilities}
    print(f"    Task Type: {task_spec.task_type}")
    print(f"    Capabilities: {task_spec.required_capabilities}")

    # 2. Extract Data
    print("\n[2] Data Extraction (T1 and T2)")
    s2_stac = Sentinel2STACDiscovery()
    
    # T1
    s2_scene_t1 = s2_stac.search_best(bbox=aoi.bbox, datetime_range=DATETIME_RANGE_T1)
    # T2
    s2_scene_t2 = s2_stac.search_best(bbox=aoi.bbox, datetime_range=DATETIME_RANGE_T2)
    
    print(f"    T1 Scene: {s2_scene_t1.item_id} ({s2_scene_t1.datetime})")
    print(f"    T2 Scene: {s2_scene_t2.item_id} ({s2_scene_t2.datetime})")

    t1_b03 = Path(tmpdir) / "T1_B03.tif"
    t2_b03 = Path(tmpdir) / "T2_B03.tif"
    
    _download_aoi_window(s2_scene_t1.assets["B03"], aoi.bbox, t1_b03)
    _download_aoi_window(s2_scene_t2.assets["B03"], aoi.bbox, t2_b03)

    # 3. Execution Engine & Change Specialist
    print("\n[3] Change Analysis (Specialist)")
    # Force the engine to use deterministic baseline for this PS workflow
    change_spec = ChangeSpecialist(default_threshold=0.10)
    ev = change_spec.infer(
        inputs=[str(t1_b03), str(t2_b03)],
        parameters={
            "t1_timestamp": s2_scene_t1.datetime,
            "t2_timestamp": s2_scene_t2.datetime
        }
    )

    registry = EvidenceRegistry()
    registry.add(ev)
    
    print(f"    Changed area: {ev.measurement.get('changed_area_km2')} km2")
    print(f"    Confidence: {ev.confidence}")

    # 4. Verification
    print("\n[4] Verification")
    verifier = GeoReasonVerifier(minimum_confidence=0.60)
    verification = verifier.verify(
        evidence=[ev],
        expected_task="temporal_analysis",
        required_modalities=["optical"]
    )

    elapsed = time.time() - start_time
    print(f"    Status: {verification.status}, Verified: {verification.verified}")
    
    print(f"\n[5] Answer Generation")
    print(f"    Analysis of {s2_scene_t1.item_id} and {s2_scene_t2.item_id} over Rasuwa.")
    print(f"    {ev.result.get('note')}. Changed Area: {ev.measurement.get('changed_area_km2')} km2.")
    print(f"    Verification status: {verification.status.upper()}")
    print(f"    Elapsed: {elapsed:.1f}s")
    
    audit = {
        "query": QUERY,
        "task_spec": task_spec.model_dump(),
        "t1_scene": s2_scene_t1.to_dict(),
        "t2_scene": s2_scene_t2.to_dict(),
        "evidence": ev.model_dump(),
        "verification": verification.model_dump()
    }
    with open(AUDIT_FILE, "w") as f:
        json.dump(audit, f, indent=2)

    return audit

if __name__ == "__main__":
    run_bi_temporal_pipeline()
