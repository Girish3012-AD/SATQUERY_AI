"""
single_image_vqa_e2e.py — Genuine single-image VQA E2E Pipeline.

Proves the full SATQuery VQA workflow using a REAL Sentinel-2 image
and genuine Qwen2-VL + SATQuery LoRA inference.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import planetary_computer as pc

from src.controller.task_controller import TaskController
from src.data.aoi_resolver import resolve_aoi, extract_location_from_query
from src.data.sentinel2_stac import Sentinel2STACDiscovery
from src.executor.vqa_specialist import VqaSpecialist, DEFAULT_ADAPTER_PATH
from src.evidence.registry import EvidenceRegistry
from src.verifier.georeason_verifier import GeoReasonVerifier

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
QUERY = "Describe the land-cover and major objects visible in this image."
DATETIME_RANGE = "2024-10-01/2024-10-31"
AOI_BBOX = [85.0, 28.0, 85.5, 28.5]  # Rasuwa, Nepal
OUTPUT_DIR = Path("outputs")
AUDIT_FILE = OUTPUT_DIR / "single_image_vqa_e2e_audit.json"


def _extract_aoi_geotiff(
    href: str, aoi_bbox: list[float], dest: Path
) -> dict[str, Any]:
    """Extract AOI window from a COG via /vsicurl/ and save as GeoTIFF."""
    signed = pc.sign(href)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="tif,tiff",
        VSI_CACHE=True,
    ):
        with rasterio.open(signed) as src:
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, *aoi_bbox
            )
            window = from_bounds(left, bottom, right, top, transform=src.transform)
            window = window.round_lengths().round_offsets()
            data = src.read(window=window)
            win_transform = src.window_transform(window)
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": window.height,
                "width": window.width,
                "transform": win_transform,
                "count": data.shape[0],
            })
            with rasterio.open(dest, "w", **meta) as dst:
                dst.write(data)

            return {
                "crs": str(src.crs),
                "bands": data.shape[0],
                "height": int(window.height),
                "width": int(window.width),
                "dtype": str(data.dtype),
            }


def run_vqa_pipeline() -> dict[str, Any]:
    """Execute the genuine single-image VQA E2E pipeline."""
    start_time = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="satquery_vqa_e2e_")
    trace: list[dict[str, Any]] = []

    def log_step(step: str, detail: Any = None):
        entry = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if detail is not None:
            entry["detail"] = detail
        trace.append(entry)
        print(f"    [{step}] {detail if detail else ''}")

    # ── 1. Query Understanding ──────────────────────────────
    print("\n[1] Query Understanding")
    log_step("query_received", QUERY)

    controller = TaskController()
    task_spec = controller.build_task_spec(query=QUERY, input_count=1)
    log_step("task_controller_routing", {
        "task_type": task_spec.task_type,
        "required_capabilities": task_spec.required_capabilities,
    })
    print(f"    Task Type: {task_spec.task_type}")
    print(f"    Capabilities: {task_spec.required_capabilities}")

    assert task_spec.task_type == "vqa", (
        f"TaskController routed to '{task_spec.task_type}' instead of 'vqa'"
    )

    # ── 2. Data Discovery & Extraction ──────────────────────
    print("\n[2] Data Discovery & Extraction (Real Sentinel-2)")
    s2_stac = Sentinel2STACDiscovery()
    s2_scene = s2_stac.search_best(bbox=AOI_BBOX, datetime_range=DATETIME_RANGE)
    log_step("scene_discovered", {
        "item_id": s2_scene.item_id,
        "datetime": str(s2_scene.datetime),
    })
    print(f"    Scene: {s2_scene.item_id}")
    print(f"    Date:  {s2_scene.datetime}")

    # Extract a 3-band GeoTIFF (B04, B03, B02 = R, G, B)
    # VqaSpecialist._prepare_model_image reads bands [3,2,1] for RGB
    geotiff_path = Path(tmpdir) / "rasuwa_RGB.tif"

    # Extract each band individually then stack
    import numpy as np
    band_data = {}
    band_meta = None
    for band_name in ["B04", "B03", "B02"]:
        signed = pc.sign(s2_scene.assets[band_name])
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS="tif,tiff",
            VSI_CACHE=True,
        ):
            with rasterio.open(signed) as src:
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326", src.crs, *AOI_BBOX
                )
                window = from_bounds(left, bottom, right, top, transform=src.transform)
                window = window.round_lengths().round_offsets()
                # Downsample to 512x512 for fast VLM inference on CPU
                band_data[band_name] = src.read(
                    1,
                    window=window,
                    out_shape=(512, 512),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
                if band_meta is None:
                    band_meta = {
                        "driver": "GTiff",
                        "height": 512,
                        "width": 512,
                        "transform": src.window_transform(window),
                        "crs": src.crs,
                        "dtype": band_data[band_name].dtype,
                        "count": 3,
                    }
        log_step(f"band_{band_name}_extracted")

    # Write stacked 3-band GeoTIFF (band order: B04=1, B03=2, B02=3)
    with rasterio.open(geotiff_path, "w", **band_meta) as dst:
        dst.write(band_data["B04"], 1)
        dst.write(band_data["B03"], 2)
        dst.write(band_data["B02"], 3)

    raster_meta = {
        "crs": str(band_meta["crs"]),
        "bands": 3,
        "height": band_meta["height"],
        "width": band_meta["width"],
        "dtype": str(band_meta["dtype"]),
    }
    log_step("image_extracted", {
        "path": str(geotiff_path),
        "assets": "B04+B03+B02 (RGB)",
        **raster_meta,
    })
    print(f"    Image: {geotiff_path.name} ({raster_meta['height']}x{raster_meta['width']}, {raster_meta['crs']})")

    # ── 3. VQA Specialist Selection ─────────────────────────
    print("\n[3] VQA Specialist Selection")
    log_step("specialist_selected", "VqaSpecialist")

    # Use the locally cached base model path for reliability
    cached_model = (
        r"C:\Users\Lenovo\.cache\huggingface\hub"
        r"\models--Qwen--Qwen2-VL-2B-Instruct"
        r"\snapshots\895c3a49bc3fa70a340399125c650a463535e71c"
    )
    if not Path(cached_model).exists():
        cached_model = "Qwen/Qwen2-VL-2B-Instruct"

    vqa_spec = VqaSpecialist(
        model_path=cached_model,
        adapter_path=DEFAULT_ADAPTER_PATH,
        unload_after_inference=True,
    )
    log_step("model_configured", {
        "base_model": cached_model,
        "adapter": DEFAULT_ADAPTER_PATH,
    })
    print(f"    Base model: Qwen2-VL-2B-Instruct")
    print(f"    Adapter:    {Path(DEFAULT_ADAPTER_PATH).name}")

    # ── 4. Real VQA Inference ───────────────────────────────
    print("\n[4] Real VQA Inference")
    log_step("inference_started")

    ev = vqa_spec.infer(
        inputs=[str(geotiff_path)],
        parameters={
            "query": QUERY,
            "sensor": "Sentinel-2",
            "modality": "optical",
            "max_new_tokens": 256,
        },
    )

    log_step("inference_completed", {
        "answer": ev.result.get("answer"),
        "evidence_id": ev.evidence_id,
        "confidence": ev.confidence,
        "confidence_method": ev.provenance.get("confidence_method"),
    })
    print(f"    Evidence ID:  {ev.evidence_id}")
    print(f"    Confidence:   {ev.confidence} (uncalibrated default)")

    # ── 5. Evidence Registration ────────────────────────────
    print("\n[5] Evidence Registration")
    registry = EvidenceRegistry()
    registry.add(ev)
    log_step("evidence_registered", {
        "evidence_id": ev.evidence_id,
        "task": ev.task,
        "modality": ev.modality,
    })

    # ── 6. Verification ────────────────────────────────────
    print("\n[6] Verification")
    verifier = GeoReasonVerifier(minimum_confidence=0.3)
    verification = verifier.verify(
        evidence=[ev],
        expected_task="vqa",
        required_modalities=["optical"],
    )
    log_step("verification_completed", {
        "status": verification.status,
        "verified": verification.verified,
        "reasons": verification.reasons,
    })
    print(f"    Status:   {verification.status}")
    print(f"    Verified: {verification.verified}")

    # ── 7. Final Response ──────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("FINAL VQA RESULT")
    print("=" * 60)
    print(f"\nQUERY: {QUERY}")
    print(f"\nMODEL OUTPUT (Qwen2-VL + SATQuery LoRA):")
    print(f"  {ev.result.get('answer')}")
    print(f"\nSYSTEM EVIDENCE:")
    print(f"  Evidence ID:       {ev.evidence_id}")
    print(f"  Task:              {ev.task}")
    print(f"  Model:             {ev.model}")
    print(f"  Sensor:            {ev.sensor}")
    print(f"  Modality:          {ev.modality}")
    print(f"  Source scene:      {s2_scene.item_id}")
    print(f"  Acquisition:       {s2_scene.datetime}")
    print(f"  Confidence:        {ev.confidence}")
    print(f"  Confidence method: {ev.provenance.get('confidence_method')}")
    print(f"  Adapter loaded:    {ev.provenance.get('adapter_loaded')}")
    print(f"  Adapter type:      {ev.provenance.get('adapter_type')}")
    print(f"  Image conversion:  {ev.provenance.get('image_input_conversion')}")
    print(f"  Device:            {ev.provenance.get('device')}")
    print(f"  Verification:      {verification.status}")
    print(f"  Elapsed:           {elapsed:.1f}s")
    print("=" * 60)

    log_step("pipeline_complete", {"elapsed_seconds": round(elapsed, 1)})

    # ── Audit file ─────────────────────────────────────────
    audit = {
        "query": QUERY,
        "task_spec": task_spec.model_dump(),
        "scene": s2_scene.to_dict(),
        "image_metadata": raster_meta,
        "evidence": ev.model_dump(),
        "verification": verification.model_dump(),
        "execution_trace": trace,
    }
    with open(AUDIT_FILE, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nAudit written to {AUDIT_FILE}")

    return audit


if __name__ == "__main__":
    run_vqa_pipeline()
