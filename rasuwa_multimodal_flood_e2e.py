"""
rasuwa_multimodal_flood_e2e.py — Genuine Optical + SAR Rasuwa E2E Pipeline.
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
from src.data.sentinel1_stac import Sentinel1STACDiscovery
import src.data.sentinel1_stac as s1_stac

from src.executor.flood_specialist import FloodSpecialist
from src.executor.sar_specialist import SARSpecialist
from src.executor.multimodal_flood_specialist import MultimodalFloodSpecialist
from src.evidence.registry import EvidenceRegistry
from src.verifier.georeason_verifier import GeoReasonVerifier

QUERY = "Identify flooded areas in Rasuwa using optical and SAR imagery."
DATETIME_RANGE = "2024-06-01/2024-10-31"
OUTPUT_DIR = Path("outputs")
AUDIT_FILE = OUTPUT_DIR / "rasuwa_multimodal_flood_e2e_audit.json"

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

def run_rasuwa_multimodal_pipeline() -> dict[str, Any]:
    start_time = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    audit: dict[str, Any] = {}
    tmpdir = tempfile.mkdtemp(prefix="satquery_multimodal_")
    
    # 1. Query & AOI
    controller = TaskController()
    task_spec = controller.build_task_spec(query=QUERY, input_count=0)
    location = extract_location_from_query(QUERY) or "Rasuwa, Nepal"
    aoi = resolve_aoi(location)
    audit["aoi"] = {"bbox": aoi.bbox}

    # 2. Optical Path
    s2_stac = Sentinel2STACDiscovery()
    s2_scene = s2_stac.search_best(bbox=aoi.bbox, datetime_range=DATETIME_RANGE)
    
    # Try reusing to speed up reruns
    b03_path = Path("C:/Users/Lenovo/AppData/Local/Temp/satquery_multimodal_ytp_whsz/B03.tif")
    b08_path = Path("C:/Users/Lenovo/AppData/Local/Temp/satquery_multimodal_ytp_whsz/B08.tif")
    
    if not b03_path.exists():
        b03_path = Path(tmpdir) / "B03.tif"
        _download_aoi_window(s2_scene.assets["B03"], aoi.bbox, b03_path)
    if not b08_path.exists():
        b08_path = Path(tmpdir) / "B08.tif"
        _download_aoi_window(s2_scene.assets["B08"], aoi.bbox, b08_path)
    
    flood_spec = FloodSpecialist()
    opt_ev = flood_spec.infer(
        inputs=[str(b03_path), str(b08_path)],
        parameters={"scene_id": s2_scene.item_id, "acquisition_date": s2_scene.datetime}
    )
    # Patch for multimodal compatibility check which requires "image_path"
    opt_ev.provenance["image_path"] = str(b03_path)

    # 3. SAR Path
    s1_stac.SENTINEL1_COLLECTION = "sentinel-1-rtc"
    s1_discovery = Sentinel1STACDiscovery()
    s1_discovery.catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    s1_scene = s1_discovery.search_best(bbox=aoi.bbox, datetime_range=DATETIME_RANGE)
    vv_path = Path("C:/Users/Lenovo/AppData/Local/Temp/satquery_multimodal_ytp_whsz/VV.tif")
    if not vv_path.exists():
        vv_path = Path(tmpdir) / "VV.tif"
        _download_aoi_window(s1_scene.assets["VV"], aoi.bbox, vv_path)
    
    sar_spec = SARSpecialist()
    sar_ev = sar_spec.infer(
        inputs=[str(vv_path)],
        parameters={"sensor": "Sentinel-1", "timestamp": s1_scene.datetime, "polarization": "VV"}
    )

    # 4. Fusion
    registry = EvidenceRegistry()
    registry.add(opt_ev)
    registry.add(sar_ev)
    
    fusion_spec = MultimodalFloodSpecialist(evidence_registry=registry)
    fused_ev = fusion_spec.infer(inputs=[opt_ev.evidence_id, sar_ev.evidence_id])
    registry.add(fused_ev)

    # 5. Verification
    verifier = GeoReasonVerifier(minimum_confidence=0.60)
    verification = verifier.verify(
        evidence=[opt_ev, sar_ev, fused_ev],
        expected_task="multimodal_flood_analysis",
        required_modalities=["optical", "sar", "optical_sar"]
    )

    elapsed = time.time() - start_time
    print(f"Status: {verification.status}, Verified: {verification.verified}")
    print(f"Elapsed: {elapsed:.1f}s")
    
    audit = {
        "s2_scene": s2_scene.item_id,
        "s1_scene": s1_scene.item_id,
        "optical_evidence": opt_ev.model_dump(),
        "sar_evidence": sar_ev.model_dump(),
        "fused_evidence": fused_ev.model_dump(),
        "verification": verification.model_dump()
    }
    with open(AUDIT_FILE, "w") as f:
        json.dump(audit, f, indent=2)

    return audit

if __name__ == "__main__":
    run_rasuwa_multimodal_pipeline()
