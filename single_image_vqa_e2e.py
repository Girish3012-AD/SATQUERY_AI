"""
single_image_vqa_e2e.py — Genuine VQA E2E Pipeline.
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
from src.executor.vqa_specialist import VqaSpecialist
from src.evidence.registry import EvidenceRegistry
from src.verifier.georeason_verifier import GeoReasonVerifier
import rasterio

QUERY = "Describe the land-cover and major objects visible in this image."
DATETIME_RANGE = "2024-10-01/2024-10-31"
OUTPUT_DIR = Path("outputs")
AUDIT_FILE = OUTPUT_DIR / "single_image_vqa_e2e_audit.json"

def _sign_href(href: str) -> str:
    import planetary_computer as pc
    return pc.sign(href)

def _download_aoi_window_as_png(href: str, aoi_bbox: list[float], dest: Path) -> Path:
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    import numpy as np
    from PIL import Image

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
            data = src.read(window=window)
            
            # Convert to RGB png
            if data.shape[0] >= 3:
                rgb = data[:3].astype(np.float32)
            else:
                rgb = np.repeat(data[0:1], 3, axis=0).astype(np.float32)
                
            valid = np.isfinite(rgb)
            if valid.any():
                low = float(np.percentile(rgb[valid], 1))
                high = float(np.percentile(rgb[valid], 99))
                rgb = np.clip((rgb - low) / (max(high - low, 1e-5)) * 255.0, 0, 255)
            
            rgb = np.transpose(rgb.astype(np.uint8), (1, 2, 0))
            Image.fromarray(rgb, mode="RGB").save(dest, format="PNG")
            
    return dest

def run_vqa_pipeline() -> dict[str, Any]:
    start_time = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    audit: dict[str, Any] = {}
    tmpdir = tempfile.mkdtemp(prefix="satquery_vqa_")
    
    print("\n[1] Query Understanding")
    controller = TaskController()
    task_spec = controller.build_task_spec(query=QUERY, input_count=1)
    location = extract_location_from_query(QUERY) or "Rasuwa, Nepal"
    aoi = resolve_aoi(location)
    audit["aoi"] = {"bbox": aoi.bbox}
    audit["task_spec"] = {"type": task_spec.task_type, "caps": task_spec.required_capabilities}
    print(f"    Task Type: {task_spec.task_type}")

    print("\n[2] Data Extraction")
    s2_stac = Sentinel2STACDiscovery()
    s2_scene = s2_stac.search_best(bbox=aoi.bbox, datetime_range=DATETIME_RANGE)
    print(f"    Scene: {s2_scene.item_id} ({s2_scene.datetime})")
    
    png_path = Path(tmpdir) / "image.png"
    _download_aoi_window_as_png(s2_scene.assets["B03"], aoi.bbox, png_path)

    print("\n[3] VQA Analysis (Specialist)")
    vqa_spec = VqaSpecialist()
    ev = vqa_spec.infer(
        inputs=[str(png_path)],
        parameters={"query": QUERY}
    )

    registry = EvidenceRegistry()
    registry.add(ev)
    
    print(f"    Answer: {ev.result.get('answer')}")
    print(f"    Confidence: {ev.confidence}")

    print("\n[4] Verification")
    verifier = GeoReasonVerifier(minimum_confidence=0.1)  # VQA might output 0.5 uncalibrated
    verification = verifier.verify(
        evidence=[ev],
        expected_task="vqa",
        required_modalities=["vqa"]
    )

    elapsed = time.time() - start_time
    print(f"    Status: {verification.status}, Verified: {verification.verified}")
    print(f"    Elapsed: {elapsed:.1f}s")
    
    audit = {
        "query": QUERY,
        "task_spec": task_spec.model_dump(),
        "scene": s2_scene.to_dict(),
        "evidence": ev.model_dump(),
        "verification": verification.model_dump()
    }
    with open(AUDIT_FILE, "w") as f:
        json.dump(audit, f, indent=2)

    return audit

if __name__ == "__main__":
    run_vqa_pipeline()
