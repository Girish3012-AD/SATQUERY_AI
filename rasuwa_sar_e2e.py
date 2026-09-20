"""
rasuwa_sar_e2e.py — End-to-end Rasuwa, Nepal SAR analysis pipeline.

Verifies real Sentinel-1 data extraction and characterization.
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
from src.data.sentinel1_stac import Sentinel1STACDiscovery
from src.executor.sar_specialist import SARSpecialist
from src.evidence.registry import EvidenceRegistry
from src.verifier.georeason_verifier import GeoReasonVerifier

QUERY = "Characterize Sentinel-1 SAR imagery in Rasuwa, Nepal."
DATETIME_RANGE = "2024-06-01/2024-10-31"
OUTPUT_DIR = Path("outputs")
AUDIT_FILE = OUTPUT_DIR / "rasuwa_sar_e2e_audit.json"

def _sign_href(href: str) -> str:
    import planetary_computer as pc
    return pc.sign(href)

def _download_aoi_window(href: str, aoi_bbox: list[float], dest: Path) -> Path:
    import rasterio
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
            
            print(f"      Requested AOI (EPSG:4326): {aoi_bbox}")
            print(f"      Projected Bounds ({src.crs}): [{left:.1f}, {bottom:.1f}, {right:.1f}, {top:.1f}]")
            print(f"      Calculated Window: {window}")
            print(f"      Output Dimensions: {window.width} x {window.height} pixels")
            
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

def run_rasuwa_sar_pipeline() -> dict[str, Any]:
    start_time = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    audit: dict[str, Any] = {}

    print("\n[1] CONTROLLER: Parsing query...")
    controller = TaskController()
    task_spec = controller.build_task_spec(query=QUERY, input_count=0)
    print(f"    task_type   : {task_spec.task_type}")
    print(f"    capabilities: {task_spec.required_capabilities}")

    print("\n[2] AOI: Resolving location from query...")
    location = extract_location_from_query(QUERY)
    if location is None:
        location = "Rasuwa, Nepal"
    aoi = resolve_aoi(location)
    print(f"    resolved    : {aoi.name}")
    print(f"    bbox        : {aoi.bbox}")
    print(f"    source      : {aoi.source}")
    audit["aoi"] = {"name": aoi.name, "bbox": aoi.bbox}

    print("\n[3] STAC: Searching Sentinel-1 scenes for Rasuwa...")
    # S1 on Planetary Computer uses sentinel-1-rtc for analysis-ready data
    stac = Sentinel1STACDiscovery()
    stac.catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    # Overwrite the constant internally if needed, but we can just use the default GRD for now since PC has it as COG.
    # Wait, Planetary computer has "sentinel-1-rtc". Let's inject it into stac.
    import src.data.sentinel1_stac as s1_stac
    s1_stac.SENTINEL1_COLLECTION = "sentinel-1-rtc"

    scene = stac.search_best(bbox=aoi.bbox, datetime_range=DATETIME_RANGE)
    print(f"    scene_id    : {scene.item_id}")
    print(f"    datetime    : {scene.datetime}")
    print(f"    collection  : {scene.collection}")
    print(f"    assets found: {list(scene.assets.keys())}")
    
    vv_href = scene.assets["VV"]
    
    tmpdir = tempfile.mkdtemp(prefix="satquery_sar_")
    vv_path = Path(tmpdir) / "VV.tif"

    print("\n[4] ASSETS: Extracting VV window...")
    print(f"    Extracting VV window from {vv_href[:60]}...")
    _download_aoi_window(vv_href, aoi.bbox, vv_path)
    print(f"    VV window saved: {vv_path.stat().st_size / 1e6:.2f} MB")

    print("\n[5] SAR ANALYSIS: Running SARSpecialist on VV...")
    specialist = SARSpecialist()
    evidence = specialist.infer(
        inputs=[str(vv_path)],
        parameters={
            "sensor": "Sentinel-1",
            "timestamp": scene.datetime,
            "polarization": "VV",
            "data_status": "real_validation"
        }
    )
    
    res = evidence.result
    print(f"    valid_pixels: {res.get('valid_pixel_count')}")
    print(f"    crs         : {res.get('crs')}")
    print(f"    width       : {res.get('width')}")
    print(f"    height      : {res.get('height')}")
    print(f"    confidence  : {evidence.confidence}")
    print(f"    evidence_id : {evidence.evidence_id}")

    print("\n[6] GEOREASON: Verifying evidence...")
    registry = EvidenceRegistry()
    registry.add(evidence)
    verifier = GeoReasonVerifier(minimum_confidence=0.60)
    verification = verifier.verify([evidence], expected_task="sar_analysis")
    print(f"    status      : {verification.status}")
    print(f"    verified    : {verification.verified}")
    print(f"    confidence  : {verification.confidence:.4f}")
    print(f"    reasons     : {verification.reasons}")

    elapsed = time.time() - start_time
    print(f"\n[7] ANSWER:\n    Sentinel-1 SAR analysis of scene {scene.item_id} (acquired {scene.datetime}) over {aoi.name}. Verification status: {verification.status.upper()}.")
    print(f"\n    Total elapsed: {elapsed:.1f}s")
    
    audit.update({
        "scene": scene.to_dict(),
        "evidence": evidence.model_dump(),
        "verification": verification.model_dump()
    })
    
    with open(AUDIT_FILE, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n[AUDIT] Saved to {AUDIT_FILE}")

    return audit

if __name__ == "__main__":
    run_rasuwa_sar_pipeline()
