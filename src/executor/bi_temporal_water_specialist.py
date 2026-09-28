from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from src.executor.specialist import Specialist
from src.schemas.evidence import Evidence
from src.geospatial.spectral import analyse_sentinel2_ndwi
from src.geospatial.polygonize import mask_to_polygons


class BiTemporalWaterChangeSpecialist(Specialist):
    """
    Quantitative Bi-Temporal Water-Area Change Specialist.

    Analyzes two co-registered Sentinel-2 images (T1 and T2) using NDWI,
    computes water area for both epochs, and calculates absolute change,
    percentage change, water loss, and water gain.
    
    Adheres strictly to scientific claim boundaries: does NOT claim
    'flood confirmed' based solely on spectral change.
    """

    CAPABILITY = "bi_temporal_water_change"
    MODEL_NAME = "BiTemporal_NDWI_WaterChange"

    REQUIRED_INPUT_PROFILE = {
        "modality": "optical",
        "temporal": "bi-temporal",
        "count": 2
    }

    DEFAULT_GREEN_BAND = 1
    DEFAULT_NIR_BAND = 4

    def __init__(
        self,
        ndwi_threshold: float = 0.0,
        min_area_px: int = 4,
    ) -> None:
        if not -1.0 <= ndwi_threshold <= 1.0:
            raise ValueError("ndwi_threshold must be in [-1.0, 1.0].")

        self.ndwi_threshold = ndwi_threshold
        self.min_area_px = min_area_px

    @property
    def capability(self) -> str:
        return self.CAPABILITY

    def _read_bands(
        self,
        path: Path,
        green_band_idx: int,
        nir_band_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, Any, str, tuple[float, float], float | None]:
        with rasterio.open(path) as ds:
            if ds.crs is None:
                raise ValueError(f"Raster has no CRS: {path}")
            
            green = ds.read(green_band_idx).astype(np.float32)
            nir = ds.read(nir_band_idx).astype(np.float32)
            transform = ds.transform
            crs = ds.crs.to_string()
            res = (abs(float(ds.transform.a)), abs(float(ds.transform.e)))
            nodata = ds.nodata
            
        return green, nir, transform, crs, res, nodata

    def _process_epoch(
        self,
        path: Path,
        green_band_idx: int,
        nir_band_idx: int,
        threshold: float,
        scene_id: str,
        acquisition_date: str,
    ) -> dict[str, Any]:
        """Process a single epoch to compute NDWI water mask and area."""
        green, nir, transform, crs, res_m, nodata = self._read_bands(
            path, green_band_idx, nir_band_idx
        )

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

        poly_result = mask_to_polygons(
            mask=spectral.mask,
            transform=transform,
            crs=crs,
            resolution_m=res_m,
            min_area_px=self.min_area_px,
        )

        return {
            "mask": spectral.mask,
            "transform": transform,
            "crs": crs,
            "resolution_m": res_m,
            "area_m2": poly_result.total_area_m2,
            "area_km2": poly_result.total_area_m2 / 1e6,
            "polygon_count": poly_result.polygon_count,
            "candidate_pixel_count": spectral.candidate_pixel_count,
            "provenance": spectral.provenance,
            "polygons": poly_result.polygons,
        }

    def infer(
        self,
        inputs: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> Evidence:
        if not inputs or len(inputs) != 2:
            raise ValueError(
                "BiTemporalWaterChangeSpecialist requires exactly two raster inputs (T1 and T2)."
            )

        parameters = parameters or {}
        start_time = time.perf_counter()

        threshold = float(parameters.get("ndwi_threshold", self.ndwi_threshold))
        
        green_band_idx = int(parameters.get("green_band", self.DEFAULT_GREEN_BAND))
        nir_band_idx = int(parameters.get("nir_band", self.DEFAULT_NIR_BAND))

        t1_path = Path(inputs[0])
        t2_path = Path(inputs[1])

        if not t1_path.exists():
            raise FileNotFoundError(f"T1 raster does not exist: {t1_path}")
        if not t2_path.exists():
            raise FileNotFoundError(f"T2 raster does not exist: {t2_path}")

        scene_id_t1 = str(parameters.get("scene_id_t1", "unknown_t1"))
        date_t1 = str(parameters.get("acquisition_date_t1", "unknown_t1"))
        
        scene_id_t2 = str(parameters.get("scene_id_t2", "unknown_t2"))
        date_t2 = str(parameters.get("acquisition_date_t2", "unknown_t2"))

        # Process T1
        t1_result = self._process_epoch(
            t1_path, green_band_idx, nir_band_idx, threshold, scene_id_t1, date_t1
        )
        
        # Process T2
        t2_result = self._process_epoch(
            t2_path, green_band_idx, nir_band_idx, threshold, scene_id_t2, date_t2
        )
        
        if t1_result["crs"] != t2_result["crs"]:
            raise ValueError("T1 and T2 rasters must have the same CRS.")
        if t1_result["resolution_m"] != t2_result["resolution_m"]:
            raise ValueError("T1 and T2 rasters must have the same resolution.")

        # Quantitative Change Metrics
        area_t1 = t1_result["area_m2"]
        area_t2 = t2_result["area_m2"]
        
        absolute_change_m2 = area_t2 - area_t1
        
        if area_t1 > 0:
            percentage_change = (absolute_change_m2 / area_t1) * 100.0
        else:
            percentage_change = float("inf") if area_t2 > 0 else 0.0
            
        water_loss_m2 = max(0.0, area_t1 - area_t2)
        water_gain_m2 = max(0.0, area_t2 - area_t1)

        # Generate Evidence ID
        hash_input = f"{scene_id_t1}:{scene_id_t2}:{area_t1}:{area_t2}:{threshold}"
        evidence_id = "CHANGE_WATER_" + hashlib.md5(hash_input.encode()).hexdigest()[:12]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        geometry_dict = None
        if t2_result["polygons"]:
            from shapely.geometry import MultiPolygon, mapping
            from src.geospatial.geometry import geometry_area
            # Show the largest polygon or all of them. Let's show all since it's water change context
            # Actually FloodSpecialist uses largest, let's use largest here to be consistent, or MultiPolygon.
            # MultiPolygon is safer to show all water.
            geometry_dict = mapping(MultiPolygon(t2_result["polygons"]))

        return Evidence(
            evidence_id=evidence_id,
            source=self.__class__.__name__,
            task=self.CAPABILITY,
            model=self.MODEL_NAME,
            sensor="Sentinel-2",
            modality="optical",
            timestamp=date_t2 if date_t2 != "unknown_t2" else None,
            geometry=geometry_dict,
            measurement={
                "area_t1_m2": area_t1,
                "area_t2_m2": area_t2,
                "area_t1_km2": t1_result["area_km2"],
                "area_t2_km2": t2_result["area_km2"],
                "absolute_change_m2": absolute_change_m2,
                "absolute_change_km2": absolute_change_m2 / 1e6,
                "percentage_change": percentage_change,
                "water_loss_m2": water_loss_m2,
                "water_loss_km2": water_loss_m2 / 1e6,
                "water_gain_m2": water_gain_m2,
                "water_gain_km2": water_gain_m2 / 1e6,
                "ndwi_threshold": threshold,
                "crs": t1_result["crs"],
                "resolution_m": t1_result["resolution_m"],
                "latency_ms": latency_ms,
            },
            result={
                "analysis_type": "bi_temporal_water_change",
                "label": "spectral_water_area_change",
                "t1_scene_id": scene_id_t1,
                "t2_scene_id": scene_id_t2,
                "absolute_change_km2": absolute_change_m2 / 1e6,
                "percentage_change": percentage_change,
                "water_gain_km2": water_gain_m2 / 1e6,
                "water_loss_km2": water_loss_m2 / 1e6,
                "note": (
                    "Spectral water-area change. Detected change in pixels classified "
                    f"as water using NDWI threshold {threshold}. Water-area change does "
                    "not by itself establish flooding, drought, construction, or causation."
                )
            },
            confidence=0.80,
            provenance={
                "t1_source": str(t1_path.resolve()),
                "t2_source": str(t2_path.resolve()),
                "t1_provenance": t1_result["provenance"],
                "t2_provenance": t2_result["provenance"],
                "model_name": self.MODEL_NAME,
                "analysis_type": "bi_temporal_ndwi_differencing",
                "confidence_calibration": "DETERMINISTIC_HEURISTIC",
            },
            metadata={
                "deterministic": True,
                "scientific_validation": False,
                "change_type": "quantitative_area",
            },
        )
