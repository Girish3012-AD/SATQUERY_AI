"""
rasuwa_flood_e2e.py — End-to-end Rasuwa, Nepal flood-analysis pipeline.

Runs the full SATQuery orchestration pipeline for the query:
  "Identify potential flooded areas in Rasuwa, Nepal using Sentinel-2 imagery."

Pipeline:
  query → TaskController → AOI resolution → Sentinel-2 STAC →
  scene selection → B03/B08 asset retrieval → FloodSpecialist (NDWI) →
  Evidence → GeoReason verification → final report

No fabricated data. Raises clear errors if real data is unavailable.
Outputs a JSON audit to outputs/rasuwa_flood_e2e_audit.json.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run as a script
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.controller.task_controller import TaskController
from src.data.aoi_resolver import resolve_aoi, extract_location_from_query
from src.data.sentinel2_stac import Sentinel2STACDiscovery
from src.executor.flood_specialist import FloodSpecialist
from src.evidence.registry import EvidenceRegistry
from src.verifier.georeason_verifier import GeoReasonVerifier


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUERY = (
    "Identify potential flooded areas in Rasuwa, Nepal "
    "using Sentinel-2 imagery."
)

# Recent date range covering monsoon/post-monsoon period
DATETIME_RANGE = "2024-06-01/2024-10-31"

OUTPUT_DIR = Path("outputs")
AUDIT_FILE = OUTPUT_DIR / "rasuwa_flood_e2e_audit.json"


def _sign_href(href: str) -> str:
    """Sign a Planetary Computer blob URL with a SAS token."""
    import planetary_computer as pc
    return pc.sign(href)


def _download_aoi_window(href: str, aoi_bbox: list[float], dest: Path) -> Path:
    """
    Extract a spatial window from a Planetary Computer COG asset matching the AOI.
    Signs the URL, uses Rasterio to read only the required pixels, and saves a local GeoTIFF.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds

    signed = _sign_href(href)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="tif",
        VSI_CACHE=True
    ):
        with rasterio.open(signed) as src:
            # aoi_bbox is [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, *aoi_bbox
            )
            
            window = from_bounds(left, bottom, right, top, transform=src.transform)
            
            # Snap the window to whole pixels to align grids properly
            window = window.round_lengths().round_offsets()
            
            print(f"      Requested AOI (EPSG:4326): {aoi_bbox}")
            print(f"      Projected Bounds ({src.crs}): [{left:.1f}, {bottom:.1f}, {right:.1f}, {top:.1f}]")
            print(f"      Calculated Window: {window}")
            print(f"      Output Dimensions: {window.width} x {window.height} pixels")
            
            # Read the data for this window
            data = src.read(1, window=window)
            
            # Calculate the geotransform for the window
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


def run_rasuwa_flood_pipeline() -> dict[str, Any]:
    """
    Execute the Rasuwa flood analysis pipeline end-to-end.
    Returns a structured audit dict.
    """
    audit: dict[str, Any] = {
        "query": QUERY,
        "pipeline": "rasuwa_flood_e2e",
        "real_data": False,
        "status": "RUNNING",
    }

    start_total = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Query understanding
    # ------------------------------------------------------------------
    print(f"\n[1] CONTROLLER: Parsing query...")
    controller = TaskController()
    task_spec = controller.build_task_spec(
        query=QUERY,
        input_count=0,
    )
    print(f"    task_type   : {task_spec.task_type}")
    print(f"    capabilities: {task_spec.required_capabilities}")

    audit["task_type"] = task_spec.task_type
    audit["detected_capabilities"] = task_spec.required_capabilities

    # ------------------------------------------------------------------
    # Step 2 — AOI resolution
    # ------------------------------------------------------------------
    print(f"\n[2] AOI: Resolving location from query...")
    location = extract_location_from_query(QUERY)
    if location is None:
        location = "Rasuwa, Nepal"   # fallback for this known query

    aoi = resolve_aoi(location)
    print(f"    resolved    : {aoi.name}")
    print(f"    bbox        : {aoi.bbox}")
    print(f"    source      : {aoi.source}")

    audit["aoi"] = {
        "name": aoi.name,
        "bbox": aoi.bbox,
        "centroid": list(aoi.centroid),
        "source": aoi.source,
    }

    # ------------------------------------------------------------------
    # Step 3 — Sentinel-2 STAC discovery
    # ------------------------------------------------------------------
    print(f"\n[3] STAC: Searching Sentinel-2 scenes for Rasuwa...")
    stac = Sentinel2STACDiscovery()

    try:
        scene = stac.search_best(
            bbox=aoi.bbox,
            datetime_range=DATETIME_RANGE,
            require_bands=("B03", "B08"),
        )
    except RuntimeError as exc:
        audit["status"] = "BLOCKED"
        audit["blocker"] = str(exc)
        print(f"    BLOCKED: {exc}")
        return audit

    print(f"    scene_id    : {scene.item_id}")
    print(f"    datetime    : {scene.datetime}")
    print(f"    cloud_cover : {scene.cloud_cover}%")
    print(f"    platform    : {scene.platform}")
    print(f"    bands found : {list(scene.assets.keys())}")

    audit["scene"] = {
        "item_id": scene.item_id,
        "collection": scene.collection,
        "datetime": scene.datetime,
        "cloud_cover_pct": scene.cloud_cover,
        "platform": scene.platform,
        "available_bands": list(scene.assets.keys()),
    }

    # ------------------------------------------------------------------
    # Step 4 — Asset retrieval (B03 + B08)
    # ------------------------------------------------------------------
    print(f"\n[4] ASSETS: Downloading B03 (Green) and B08 (NIR)...")

    tmpdir = tempfile.mkdtemp(prefix="satquery_rasuwa_")
    b03_href = scene.assets.get("B03")
    b08_href = scene.assets.get("B08")

    if not b03_href or not b08_href:
        audit["status"] = "BLOCKED"
        audit["blocker"] = (
            f"Required bands B03/B08 not in scene assets. "
            f"Found: {list(scene.assets.keys())}"
        )
        print(f"    BLOCKED: {audit['blocker']}")
        return audit

    b03_path = Path(tmpdir) / "B03_green.tif"
    b08_path = Path(tmpdir) / "B08_nir.tif"

    print(f"    Extracting B03 window from {b03_href[:60]}...")
    _download_aoi_window(b03_href, aoi.bbox, b03_path)
    print(f"    B03 window saved: {b03_path.stat().st_size / 1e6:.2f} MB")

    print(f"    Extracting B08 window from {b08_href[:60]}...")
    _download_aoi_window(b08_href, aoi.bbox, b08_path)
    print(f"    B08 window saved: {b08_path.stat().st_size / 1e6:.2f} MB")

    audit["assets"] = {
        "B03_href": b03_href,
        "B08_href": b08_href,
        "B03_local": str(b03_path),
        "B08_local": str(b08_path),
        "B03_size_mb": round(b03_path.stat().st_size / 1e6, 2),
        "B08_size_mb": round(b08_path.stat().st_size / 1e6, 2),
    }
    audit["real_data"] = True

    # ------------------------------------------------------------------
    # Step 5 — FloodSpecialist (NDWI)
    # ------------------------------------------------------------------
    print(f"\n[5] FLOOD ANALYSIS: Running NDWI on B03+B08...")

    evidence_registry = EvidenceRegistry()
    specialist = FloodSpecialist(ndwi_threshold=0.0, min_area_px=10)

    evidence = specialist.infer(
        [str(b03_path), str(b08_path)],
        parameters={
            "scene_id": scene.item_id,
            "acquisition_date": scene.datetime or "unknown",
            "ndwi_threshold": 0.0,
        },
    )

    evidence_registry.add(evidence)

    m = evidence.measurement
    print(f"    evidence_id         : {evidence.evidence_id}")
    print(f"    candidate_pixels    : {m['candidate_pixel_count']:,}")
    print(f"    candidate_area_km2  : {m['total_candidate_area_km2']:.4f}")
    print(f"    polygon_count       : {m['candidate_polygon_count']}")
    print(f"    confidence          : {evidence.confidence}")
    print(f"    label               : {evidence.result['label']}")

    audit["ndwi_analysis"] = {
        "evidence_id": evidence.evidence_id,
        "threshold": m["ndwi_threshold"],
        "candidate_pixel_count": m["candidate_pixel_count"],
        "nodata_pixel_count": m["nodata_pixel_count"],
        "candidate_polygon_count": m["candidate_polygon_count"],
        "candidate_area_m2": m["total_candidate_area_m2"],
        "candidate_area_km2": m["total_candidate_area_km2"],
        "resolution_m": list(m["resolution_m"]),
        "label": evidence.result["label"],
        "confidence": evidence.confidence,
        "note": evidence.result["note"],
    }

    # ------------------------------------------------------------------
    # Step 6 — GeoReason verification
    # ------------------------------------------------------------------
    print(f"\n[6] GEOREASON: Verifying evidence...")

    verifier = GeoReasonVerifier(minimum_confidence=0.50)
    verification = verifier.verify(
        evidence_registry.all(),
        expected_task="flood_detection",
    )

    print(f"    status              : {verification.status}")
    print(f"    verified            : {verification.verified}")
    print(f"    confidence          : {verification.confidence:.4f}")
    print(f"    reasons             : {verification.reasons}")

    audit["verification"] = {
        "status": verification.status,
        "verified": verification.verified,
        "confidence": verification.confidence,
        "reasons": verification.reasons,
        "conflicts": verification.conflicts,
        "recommended_action": verification.recommended_action,
    }

    # ------------------------------------------------------------------
    # Step 7 — Final answer
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - start_total
    area_km2 = m["total_candidate_area_km2"]
    polygon_count = m["candidate_polygon_count"]

    answer = (
        f"Sentinel-2 NDWI analysis of scene {scene.item_id} "
        f"(acquired {scene.datetime}, cloud cover {scene.cloud_cover}%) "
        f"over Rasuwa District, Nepal detected "
        f"{polygon_count} potential water/flood-candidate region(s) "
        f"covering approximately {area_km2:.4f} km². "
        f"Verification status: {verification.status.upper()}. "
        f"IMPORTANT: This result represents spectral water similarity "
        f"(NDWI >= 0.0) and is NOT confirmed flooding."
    )

    print(f"\n[7] ANSWER:")
    print(f"    {answer}")
    print(f"\n    Total elapsed: {elapsed:.1f}s")

    audit["answer"] = answer
    audit["elapsed_seconds"] = round(elapsed, 1)
    audit["status"] = "COMPLETE"
    audit["scientific_validation"] = False

    # ------------------------------------------------------------------
    # Save audit
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.write_text(
        json.dumps(audit, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[AUDIT] Saved to {AUDIT_FILE}")

    return audit


if __name__ == "__main__":
    result = run_rasuwa_flood_pipeline()
    status = result.get("status", "UNKNOWN")
    print(f"\n{'='*60}")
    print(f"PIPELINE STATUS: {status}")
    if status == "BLOCKED":
        print(f"BLOCKER: {result.get('blocker')}")
        sys.exit(1)
