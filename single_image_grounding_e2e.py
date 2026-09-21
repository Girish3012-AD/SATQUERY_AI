"""
single_image_grounding_e2e.py — Genuine single-image Spatial Grounding E2E Pipeline.

Proves the SATQuery spatial grounding workflow using a REAL Sentinel-2 image
and deterministic NDWI spectral detection + polygonization.
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

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import planetary_computer as pc

from src.controller.task_controller import TaskController
from src.data.aoi_resolver import resolve_aoi, extract_location_from_query
from src.data.sentinel2_stac import Sentinel2STACDiscovery
from src.executor.water_specialist import WaterSpecialist
from src.evidence.registry import EvidenceRegistry
from src.verifier.georeason_verifier import GeoReasonVerifier

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
QUERY = "Highlight the water body referred to in the image."
DATETIME_RANGE = "2024-10-01/2024-10-31"
AOI_BBOX = [85.0, 28.0, 85.5, 28.5]  # Rasuwa, Nepal
OUTPUT_DIR = Path("outputs")
AUDIT_FILE = OUTPUT_DIR / "single_image_grounding_e2e_audit.json"
OVERLAY_FILE = OUTPUT_DIR / "grounding_water_overlay.png"


def create_visual_overlay(
    b04_data: np.ndarray,
    b03_data: np.ndarray,
    b02_data: np.ndarray,
    water_mask: np.ndarray,
    output_path: Path,
) -> Path:
    """
    Create a visual grounding overlay artifact:
    Real satellite RGB image + semi-transparent blue highlight on grounded water pixels.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize RGB channels (1st to 99th percentile)
    rgb = np.stack([b04_data, b03_data, b02_data], axis=-1).astype(np.float32)
    valid = np.isfinite(rgb)
    if valid.any():
        low = float(np.percentile(rgb[valid], 1))
        high = float(np.percentile(rgb[valid], 99))
        rgb = np.clip((rgb - low) / (max(high - low, 1e-5)) * 255.0, 0, 255)
    rgb_uint8 = rgb.astype(np.uint8)

    base_img = Image.fromarray(rgb_uint8, mode="RGB").convert("RGBA")

    # Create RGBA overlay: blue highlight for water candidate mask
    overlay = np.zeros((water_mask.shape[0], water_mask.shape[1], 4), dtype=np.uint8)

    # Water pixels: cyan-blue fill (0, 150, 255, alpha=130)
    overlay[water_mask] = [0, 150, 255, 130]

    # Draw border around water regions
    from scipy.ndimage import binary_dilation
    border = water_mask & ~binary_dilation(water_mask, iterations=1)
    overlay[border] = [0, 255, 255, 255]  # bright cyan border

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    combined = Image.alpha_composite(base_img, overlay_img)
    combined.convert("RGB").save(output_path, format="PNG")

    return output_path


def run_grounding_pipeline() -> dict[str, Any]:
    """Execute the genuine single-image spatial grounding E2E pipeline."""
    start_time = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="satquery_grounding_e2e_")
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

    # ── 1. Query Received & TaskController Routing ──────────
    print("\n[1] Query Understanding & TaskController Routing")
    log_step("query_received", QUERY)

    controller = TaskController()
    task_spec = controller.build_task_spec(query=QUERY, input_count=1)
    log_step("task_controller_routing", {
        "task_type": task_spec.task_type,
        "required_capabilities": task_spec.required_capabilities,
        "spatial_operations": task_spec.spatial_operations,
    })
    print(f"    Task Type:             {task_spec.task_type}")
    print(f"    Required Capabilities: {task_spec.required_capabilities}")

    # ── 2. Scene Discovery & Raster Extraction ───────────────
    print("\n[2] Scene Discovery & Raster Extraction (Real Sentinel-2)")
    s2_stac = Sentinel2STACDiscovery()
    s2_scene = s2_stac.search_best(bbox=AOI_BBOX, datetime_range=DATETIME_RANGE)
    log_step("scene_discovered", {
        "item_id": s2_scene.item_id,
        "datetime": str(s2_scene.datetime),
    })
    print(f"    Scene: {s2_scene.item_id}")
    print(f"    Date:  {s2_scene.datetime}")

    # Extract B03 (Green) and B08 (NIR) single-band GeoTIFFs
    b03_path = Path(tmpdir) / "B03.tif"
    b08_path = Path(tmpdir) / "B08.tif"
    b04_path = Path(tmpdir) / "B04.tif"
    b02_path = Path(tmpdir) / "B02.tif"

    extracted_data = {}
    raster_meta = None

    for bname, bpath in [
        ("B03", b03_path),
        ("B08", b08_path),
        ("B04", b04_path),
        ("B02", b02_path),
    ]:
        signed = pc.sign(s2_scene.assets[bname])
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
                arr = src.read(
                    1,
                    window=window,
                    out_shape=(512, 512),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
                win_transform = src.window_transform(window)
                extracted_data[bname] = arr
                if raster_meta is None:
                    raster_meta = {
                        "driver": "GTiff",
                        "height": 512,
                        "width": 512,
                        "transform": win_transform,
                        "crs": str(src.crs),
                        "dtype": str(arr.dtype),
                    }
                meta = src.meta.copy()
                meta.update({
                    "driver": "GTiff",
                    "height": 512,
                    "width": 512,
                    "transform": win_transform,
                    "count": 1,
                })
                with rasterio.open(bpath, "w", **meta) as dst:
                    dst.write(arr, 1)

        log_step(f"band_{bname}_extracted", {"path": str(bpath)})

    log_step("image_extracted", {
        "bands": list(extracted_data.keys()),
        **raster_meta,
    })
    print(f"    Rasters extracted: 512x512 ({raster_meta['crs']})")

    # ── 3. Grounding Specialist Selection ───────────────────
    print("\n[3] Grounding Specialist Selection")
    log_step("grounding_specialist_selected", "WaterSpecialist")

    specialist = WaterSpecialist(ndwi_threshold=0.0, min_area_px=4)
    print(f"    Specialist: {specialist.capability} ({specialist.MODEL_NAME})")

    # ── 4. Grounding Inference & Spatial Analysis ───────────
    print("\n[4] Grounding Inference & Spatial Analysis")
    log_step("grounding_inference_started")

    ev = specialist.infer(
        inputs=[str(b03_path), str(b08_path)],
        parameters={
            "scene_id": s2_scene.item_id,
            "acquisition_date": str(s2_scene.datetime),
            "ndwi_threshold": 0.0,
        },
    )

    log_step("grounding_result_generated", {
        "evidence_id": ev.evidence_id,
        "has_geometry": ev.geometry is not None,
        "water_polygon_count": ev.measurement.get("water_polygon_count"),
        "total_water_area_km2": ev.measurement.get("total_water_area_km2"),
        "bbox_geographic": ev.result.get("bbox_geographic"),
        "grounding_method": ev.result.get("grounding_method"),
    })

    print(f"    Evidence ID:          {ev.evidence_id}")
    print(f"    Grounded Polygons:    {ev.measurement.get('water_polygon_count')}")
    print(f"    Total Grounded Area:  {ev.measurement.get('total_water_area_km2'):.4f} km²")
    print(f"    Geographic Bounding Box: {ev.result.get('bbox_geographic')}")

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
    verifier = GeoReasonVerifier(minimum_confidence=0.5)
    verification = verifier.verify(
        evidence=[ev],
        expected_task="water_detection",
        required_modalities=["optical"],
    )
    log_step("verification_completed", {
        "status": verification.status,
        "verified": verification.verified,
        "reasons": verification.reasons,
    })
    print(f"    Status:   {verification.status}")
    print(f"    Verified: {verification.verified}")

    # ── 7. Visual Evidence Overlay Generation ───────────────
    print("\n[7] Visual Evidence Overlay Generation")
    overlay_path = create_visual_overlay(
        b04_data=extracted_data["B04"],
        b03_data=extracted_data["B03"],
        b02_data=extracted_data["B02"],
        water_mask=specialist.last_mask,
        output_path=OVERLAY_FILE,
    )
    log_step("visual_evidence_generated", {"overlay_path": str(overlay_path)})
    print(f"    Visual overlay written to {overlay_path}")

    # ── 8. Final Response & Summary ──────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("FINAL GROUNDING RESULT")
    print("=" * 60)
    print(f"\nQUERY: {QUERY}")
    print(f"\nGROUNDING METHOD:")
    print(f"  Geospatial Raster Spectral Grounding (NDWI >= 0.0 + Polygonization)")
    print(f"  Note: Deterministic remote-sensing spectral detection of water bodies.")
    print(f"\nSPATIAL GROUNDING OUTPUT:")
    print(f"  Grounded Feature:    Water Body (Candidate Polygons)")
    print(f"  Polygon Count:       {ev.measurement.get('water_polygon_count')}")
    print(f"  Total Grounded Area: {ev.measurement.get('total_water_area_km2'):.4f} km² ({ev.measurement.get('total_water_area_m2'):.1f} m²)")
    print(f"  Bounding Box (CRS):  {ev.result.get('bbox_geographic')}")
    print(f"  CRS:                 {ev.result.get('crs')}")
    print(f"\nSYSTEM EVIDENCE:")
    print(f"  Evidence ID:         {ev.evidence_id}")
    print(f"  Task:                {ev.task}")
    print(f"  Model:               {ev.model}")
    print(f"  Sensor:              {ev.sensor}")
    print(f"  Source Scene:        {s2_scene.item_id}")
    print(f"  Acquisition Date:    {s2_scene.datetime}")
    print(f"  Confidence:          {ev.confidence} (uncalibrated default)")
    print(f"  Verification Status: {verification.status}")
    print(f"  Visual Overlay:      {overlay_path}")
    print(f"  Elapsed Time:        {elapsed:.1f}s")
    print("=" * 60)

    log_step("pipeline_complete", {"elapsed_seconds": round(elapsed, 1)})

    # Audit file
    audit = {
        "query": QUERY,
        "task_spec": task_spec.model_dump(),
        "scene": s2_scene.to_dict(),
        "image_metadata": raster_meta,
        "evidence": ev.model_dump(),
        "verification": verification.model_dump(),
        "visual_overlay_path": str(overlay_path),
        "execution_trace": trace,
    }
    with open(AUDIT_FILE, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nAudit written to {AUDIT_FILE}")

    return audit


if __name__ == "__main__":
    run_grounding_pipeline()
