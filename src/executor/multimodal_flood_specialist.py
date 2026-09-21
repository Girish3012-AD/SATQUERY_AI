from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import numpy as np
import rasterio
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union

from src.executor.specialist import Specialist
from src.schemas.evidence import Evidence
from src.evidence.registry import EvidenceRegistry
from src.geospatial.multimodal_alignment import validate_optical_sar_evidence_compatibility
from src.geospatial.geometry import geometry_area
from src.geospatial.polygonize import mask_to_polygons
from src.geospatial.spectral import analyse_sentinel2_ndwi


class MultimodalFloodSpecialist(Specialist):
    """
    Genuine decision/feature-level fusion for Optical + SAR flood analysis.
    
    Extracts deterministic optical NDWI flood candidates and intersects them 
    with deterministic SAR low-backscatter candidates.
    """

    CAPABILITY = "multimodal_flood_analysis"
    MODEL_NAME = "OpticalSAR_Deterministic_Fusion"

    def __init__(self, evidence_registry: EvidenceRegistry | None = None) -> None:
        self.evidence_registry = evidence_registry or EvidenceRegistry()
        # Planetary Computer Sentinel-1 RTC is in linear power scale.
        # -15 dB is approximately 0.0316 in linear power.
        self.sar_water_threshold = 0.0316

    @property
    def capability(self) -> str:
        return self.CAPABILITY

    def infer(
        self,
        inputs: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> Evidence:
        start_time = time.perf_counter()

        if len(inputs) != 2:
            raise ValueError(
                "MultimodalFloodSpecialist requires exactly two evidence IDs "
                "(optical and SAR)."
            )

        ev1 = self.evidence_registry.get(inputs[0])
        ev2 = self.evidence_registry.get(inputs[1])

        if ev1.modality == "optical" and ev2.modality == "sar":
            opt_ev, sar_ev = ev1, ev2
        elif ev2.modality == "optical" and ev1.modality == "sar":
            opt_ev, sar_ev = ev2, ev1
        else:
            raise ValueError("Required one optical and one sar evidence.")

        evidence_id = f"FUSE_{uuid4().hex[:8]}"

        compat = validate_optical_sar_evidence_compatibility(
            opt_ev, 
            sar_ev, 
            max_temporal_separation_seconds=9999999.0  # Allow large temporal gaps for demonstration
        )
        if compat["status"] != "compatible" or not compat["spatial"]["crs_compatible"]:
            return self._abstain(
                evidence_id=evidence_id,
                opt_ev=opt_ev,
                sar_ev=sar_ev,
                compat=compat,
                reason="Optical and SAR evidence are not spatially/temporally compatible."
            )

        opt_path = opt_ev.provenance.get("source_path")
        if not opt_path:
            opt_path = opt_ev.provenance.get("image_path")
            
        sar_path = sar_ev.provenance.get("image_path")

        if not opt_path or not sar_path:
            return self._abstain(
                evidence_id=evidence_id,
                opt_ev=opt_ev,
                sar_ev=sar_ev,
                compat=compat,
                reason="Missing raster paths in provenance."
            )

        # 1. Optical Branch
        if "," in opt_path:
            p1, p2 = opt_path.split(",")
            with rasterio.open(p1) as src1:
                opt_transform = src1.transform
                opt_crs = src1.crs
                opt_res = src1.res
                green = src1.read(1)
            with rasterio.open(p2) as src2:
                nir = src2.read(1)
        else:
            with rasterio.open(opt_path) as src_opt:
                opt_transform = src_opt.transform
                opt_crs = src_opt.crs
                opt_res = src_opt.res
                green = src_opt.read(1)
                nir = src_opt.read(2)

        spectral_result = analyse_sentinel2_ndwi(
            green_band=green,
            nir_band=nir,
            transform=opt_transform,
            crs=str(opt_crs),
            resolution_m=opt_res
        )
        opt_poly_result = mask_to_polygons(
            mask=spectral_result.mask,
            transform=opt_transform,
            crs=str(opt_crs)
        )

        # 2. SAR Branch
        with rasterio.open(sar_path) as src_sar:
            sar_transform = src_sar.transform
            sar_crs = src_sar.crs
            sar_data = src_sar.read(1)
            sar_nodata = src_sar.nodata

        sar_mask = (sar_data < self.sar_water_threshold)
        if sar_nodata is not None:
            sar_mask = sar_mask & (sar_data != sar_nodata)

        sar_poly_result = mask_to_polygons(
            mask=sar_mask,
            transform=sar_transform,
            crs=str(sar_crs)
        )

        # 3. Fusion (Intersection)
        sar_union = unary_union(sar_poly_result.polygons) if sar_poly_result.polygons else Polygon()
        
        fused_polygons = []
        for p in opt_poly_result.polygons:
            intersection = p.intersection(sar_union)
            if not intersection.is_empty:
                if isinstance(intersection, Polygon):
                    fused_polygons.append(intersection)
                elif isinstance(intersection, MultiPolygon):
                    fused_polygons.extend(list(intersection.geoms))

        total_area = sum(geometry_area(p) for p in fused_polygons)
        
        geometry_dict = None
        if fused_polygons:
            largest = max(fused_polygons, key=geometry_area)
            geometry_dict = largest.__geo_interface__

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        provenance = {
            "optical_evidence_id": opt_ev.evidence_id,
            "sar_evidence_id": sar_ev.evidence_id,
            "optical_source": opt_path,
            "sar_source": sar_path,
            "optical_date": opt_ev.timestamp,
            "sar_date": sar_ev.timestamp,
            "fusion_rule": "optical_ndwi_intersect_sar_low_backscatter",
            "sar_threshold_db": self.sar_water_threshold,
            "model_name": self.MODEL_NAME,
            "analysis_type": "multimodal_flood_fusion",
            "confidence_calibration": "DETERMINISTIC_HEURISTIC"
        }

        measurement = {
            "optical_candidate_polygons": opt_poly_result.polygon_count,
            "sar_candidate_polygons": sar_poly_result.polygon_count,
            "fused_candidate_polygons": len(fused_polygons),
            "fused_area_km2": total_area / 1e6,
            "crs": str(opt_crs),
            "latency_ms": latency_ms
        }

        result = {
            "analysis_type": "multimodal_flood_candidate",
            "status": "success",
            "fused_area_km2": total_area / 1e6,
            "polygon_count": len(fused_polygons),
            "spatial_compatibility": compat["spatial"],
            "note": "multimodal flood candidate/supporting evidence. Derived from deterministic NDWI and SAR threshold intersection."
        }

        # Confidence is 0.85 if fused area is found, else 0.4
        confidence = 0.85 if total_area > 0 else 0.40

        return Evidence(
            evidence_id=evidence_id,
            source=self.__class__.__name__,
            task=self.CAPABILITY,
            model=self.MODEL_NAME,
            sensor="optical_sar",
            modality="optical_sar",
            timestamp=opt_ev.timestamp,
            geometry=geometry_dict,
            measurement=measurement,
            result=result,
            confidence=confidence,
            provenance=provenance,
            metadata={"deterministic": True}
        )

    def _abstain(
        self,
        evidence_id: str,
        opt_ev: Evidence,
        sar_ev: Evidence,
        compat: dict[str, Any],
        reason: str
    ) -> Evidence:
        return Evidence(
            evidence_id=evidence_id,
            source=self.__class__.__name__,
            task=self.CAPABILITY,
            model=self.MODEL_NAME,
            sensor="optical_sar",
            modality="optical_sar",
            timestamp=opt_ev.timestamp,
            geometry=None,
            measurement={"status": "abstained", "reason": reason},
            result={"status": "abstained", "reason": reason, "compatibility": compat},
            confidence=0.0,
            provenance={
                "optical_evidence_id": opt_ev.evidence_id,
                "sar_evidence_id": sar_ev.evidence_id,
            },
            metadata={"deterministic": True}
        )
