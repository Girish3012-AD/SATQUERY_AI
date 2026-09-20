"""
flood_specialist.py — Sentinel-2 NDWI-based potential-flood specialist.

Integration layer for SATQuery's flood_detection capability.

Pipeline (all deterministic — no LLM):
  GeoTIFF (B03 Green + B08 NIR)
    → NDWI (spectral.py)
    → water candidate mask
    → geographic polygons (polygonize.py)
    → area statistics (geometry.py)
    → Evidence (schemas/evidence.py)

Label convention:
  "potential_water_candidate" — NOT "confirmed flooding".
  NDWI threshold detects spectral water similarity; validated
  flood detection would require a trained classifier.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from src.schemas.evidence import Evidence

from .specialist import Specialist
from src.geospatial.spectral import analyse_sentinel2_ndwi
from src.geospatial.polygonize import mask_to_polygons
from src.geospatial.geometry import geometry_area


class FloodSpecialist(Specialist):
    """
    Deterministic potential-flood/water detection specialist.

    Uses NDWI on Sentinel-2 B03 (Green) + B08 (NIR) bands.
    Input must be a multi-band GeoTIFF with B03 and B08 on
    separate bands, OR two single-band GeoTIFFs (B03 first, B08 second).

    Produces SATQuery Evidence with:
      - candidate polygon geometry (largest contiguous region)
      - area statistics
      - full provenance
    """

    CAPABILITY = "flood_detection"
    MODEL_NAME = "NDWI_Sentinel2_FloodCandidate"

    # Band layout when reading a multi-band GeoTIFF.
    # Overridable via infer() parameters.
    DEFAULT_GREEN_BAND = 1   # 1-indexed (B03 = band 1 in standard 4-band)
    DEFAULT_NIR_BAND = 4     # B08 = band 4 in standard BGRN ordering

    # Single-band input mode band indices
    GREEN_BAND_INDEX = 1
    NIR_BAND_INDEX = 1

    def __init__(
        self,
        ndwi_threshold: float = 0.0,
        min_area_px: int = 4,
    ) -> None:
        if not -1.0 <= ndwi_threshold <= 1.0:
            raise ValueError(
                "ndwi_threshold must be in [-1.0, 1.0]."
            )

        self.ndwi_threshold = ndwi_threshold
        self.min_area_px = min_area_px

    @property
    def capability(self) -> str:
        return self.CAPABILITY

    # ------------------------------------------------------------------
    # Band reading helpers
    # ------------------------------------------------------------------

    def _read_single_band(
        self,
        path: Path,
        band_index: int = 1,
    ) -> tuple[np.ndarray, Any, str, tuple[float, float], float | None]:
        """Read one band from a GeoTIFF. Returns (array, transform, crs, res, nodata)."""
        with rasterio.open(path) as ds:
            if ds.crs is None:
                raise ValueError(
                    f"Raster has no CRS: {path}"
                )
            data = ds.read(band_index).astype(np.float32)
            transform = ds.transform
            crs = ds.crs.to_string()
            res = (abs(float(ds.transform.a)), abs(float(ds.transform.e)))
            nodata = ds.nodata
        return data, transform, crs, res, nodata

    def _read_two_bands(
        self,
        path: Path,
        green_band: int,
        nir_band: int,
    ) -> tuple[np.ndarray, np.ndarray, Any, str, tuple[float, float], float | None]:
        """Read two bands from the same GeoTIFF."""
        with rasterio.open(path) as ds:
            if ds.crs is None:
                raise ValueError(f"Raster has no CRS: {path}")
            if ds.count < max(green_band, nir_band):
                raise ValueError(
                    f"Raster has {ds.count} bands; "
                    f"requested green={green_band}, nir={nir_band}."
                )
            green = ds.read(green_band).astype(np.float32)
            nir = ds.read(nir_band).astype(np.float32)
            transform = ds.transform
            crs = ds.crs.to_string()
            res = (abs(float(ds.transform.a)), abs(float(ds.transform.e)))
            nodata = ds.nodata
        return green, nir, transform, crs, res, nodata

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def infer(
        self,
        inputs: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> Evidence:
        """
        Run the NDWI flood-candidate analysis.

        Inputs
        ------
        inputs[0] : path to a multi-band GeoTIFF (B03/B08 on separate bands)
                    OR path to B03 (Green) single-band GeoTIFF
        inputs[1] : (optional) path to B08 (NIR) single-band GeoTIFF

        Parameters
        ----------
        green_band      : band index for green (default 1, 1-indexed)
        nir_band        : band index for NIR (default 4, 1-indexed)
        ndwi_threshold  : override instance threshold
        scene_id        : provenance scene ID
        acquisition_date: provenance date
        """
        if not inputs:
            raise ValueError(
                "FloodSpecialist requires at least one raster input."
            )

        parameters = parameters or {}
        start = time.perf_counter()

        threshold = float(
            parameters.get("ndwi_threshold", self.ndwi_threshold)
        )
        scene_id = str(parameters.get("scene_id", "unknown"))
        acquisition_date = str(parameters.get("acquisition_date", "unknown"))

        # ------------------------------------------------------------------
        # Band loading
        # ------------------------------------------------------------------
        if len(inputs) == 1:
            # Single multi-band file
            path = Path(inputs[0])
            if not path.exists():
                raise FileNotFoundError(
                    f"Input raster does not exist: {path}"
                )

            green_band = int(parameters.get(
                "green_band", self.DEFAULT_GREEN_BAND
            ))
            nir_band = int(parameters.get(
                "nir_band", self.DEFAULT_NIR_BAND
            ))

            green, nir, transform, crs, res_m, nodata = (
                self._read_two_bands(path, green_band, nir_band)
            )
            source_path = str(path.resolve())

        else:
            # Two single-band files: inputs[0]=B03, inputs[1]=B08
            b03_path = Path(inputs[0])
            b08_path = Path(inputs[1])

            for p in (b03_path, b08_path):
                if not p.exists():
                    raise FileNotFoundError(
                        f"Input raster does not exist: {p}"
                    )

            green, transform, crs, res_m, nodata = (
                self._read_single_band(b03_path, self.GREEN_BAND_INDEX)
            )
            nir, _, _, _, _ = (
                self._read_single_band(b08_path, self.NIR_BAND_INDEX)
            )
            source_path = f"{b03_path.resolve()},{b08_path.resolve()}"

        # ------------------------------------------------------------------
        # NDWI + mask
        # ------------------------------------------------------------------
        spectral = analyse_sentinel2_ndwi(
            green_band=green,
            nir_band=nir,
            transform=transform,
            crs=crs,
            resolution_m=res_m,
            nodata=nodata,
            threshold=threshold,
            scene_id=scene_id,
            acquisition_date=acquisition_date,
        )

        # ------------------------------------------------------------------
        # Polygonize
        # ------------------------------------------------------------------
        poly_result = mask_to_polygons(
            mask=spectral.mask,
            transform=transform,
            crs=crs,
            resolution_m=res_m,
            min_area_px=self.min_area_px,
        )

        latency_ms = (time.perf_counter() - start) * 1000.0

        # ------------------------------------------------------------------
        # Evidence geometry — use the largest polygon or union if multiple
        # ------------------------------------------------------------------
        geometry_dict: dict[str, Any] | None = None
        if poly_result.polygons:
            largest = max(poly_result.polygons, key=geometry_area)
            geometry_dict = largest.__geo_interface__

        # ------------------------------------------------------------------
        # Evidence ID — deterministic hash of key statistics
        # ------------------------------------------------------------------
        hash_input = (
            f"{scene_id}:{spectral.candidate_pixel_count}:"
            f"{poly_result.total_area_m2:.2f}:{threshold}"
        )
        evidence_id = (
            "FLOOD_NDWI_"
            + hashlib.md5(hash_input.encode()).hexdigest()[:12]
        )

        # ------------------------------------------------------------------
        # Build Evidence
        # ------------------------------------------------------------------
        return Evidence(
            evidence_id=evidence_id,
            source="FloodSpecialist",
            task=self.CAPABILITY,
            model=self.MODEL_NAME,
            sensor="Sentinel-2",
            modality="optical",
            timestamp=acquisition_date if acquisition_date != "unknown" else None,
            geometry=geometry_dict,
            measurement={
                "candidate_pixel_count": spectral.candidate_pixel_count,
                "nodata_pixel_count": spectral.nodata_pixel_count,
                "candidate_polygon_count": poly_result.polygon_count,
                "total_candidate_area_m2": poly_result.total_area_m2,
                "total_candidate_area_km2": (
                    poly_result.total_area_m2 / 1e6
                ),
                "ndwi_threshold": threshold,
                "resolution_m": res_m,
                "latency_ms": latency_ms,
            },
            result={
                "analysis_type": "ndwi_water_candidate",
                "label": "potential_water_candidate",
                "scene_id": scene_id,
                "acquisition_date": acquisition_date,
                "crs": crs,
                "candidate_area_km2": poly_result.total_area_m2 / 1e6,
                "polygon_count": poly_result.polygon_count,
                "ndwi_threshold": threshold,
                "note": (
                    "NDWI-based spectral detection. Result represents "
                    "potential water/flood-affected areas. "
                    "NOT confirmed flooding."
                ),
            },
            confidence=0.70,   # Uncalibrated NDWI threshold — honest estimate
            provenance=spectral.provenance | {
                "source_path": source_path,
                "polygonization": poly_result.provenance,
                "model_name": self.MODEL_NAME,
                "analysis_type": "ndwi_spectral_detection",
                "confidence_calibration": "NOT_CALIBRATED",
            },
            metadata={
                "deterministic": True,
                "scientific_validation": False,
                "warning": (
                    "NDWI threshold detects spectral water similarity. "
                    "This is not a validated flood classifier."
                ),
            },
        )
