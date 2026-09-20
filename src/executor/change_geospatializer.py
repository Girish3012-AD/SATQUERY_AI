from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

from src.schemas.evidence import Evidence


class ChangeGeospatializer:
    """Convert a temporal change mask into geometry-bearing Evidence."""

    TASK = "temporal_change_geospatialization"
    SOURCE = "change_mask_geospatializer"
    MODEL = "deterministic-change-geospatializer"

    def __init__(
        self,
        output_dir: str = "outputs/sentinel2_temporal_change",
    ) -> None:
        self.output_dir = Path(output_dir)

    @staticmethod
    def _make_artifact_id(source_evidence: Evidence) -> str:
        payload = (
            f"{source_evidence.evidence_id}|"
            "change_mask_geospatializer"
        )
        digest = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:12]
        return f"CHG_GEO_{digest}"

    def _write_mask(
        self,
        mask: np.ndarray,
        transform: Any,
        crs: Any,
        output_path: Path,
    ) -> None:
        if mask.ndim != 2:
            raise ValueError(
                f"Change mask must be 2D, got {mask.shape}."
            )

        if crs is None:
            raise ValueError(
                "Change mask requires a valid CRS."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_uint8 = np.asarray(
            mask,
            dtype=np.uint8,
        )

        profile = {
            "driver": "GTiff",
            "height": int(mask_uint8.shape[0]),
            "width": int(mask_uint8.shape[1]),
            "count": 1,
            "dtype": "uint8",
            "crs": crs,
            "transform": transform,
            "nodata": None,
            "compress": "lzw",
        }

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:
            dst.write(mask_uint8, 1)

    @staticmethod
    def _polygonize_mask(mask_path: Path):
        with rasterio.open(mask_path) as dataset:
            mask = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs

            if crs is None:
                raise ValueError(
                    "Change mask GeoTIFF contains no CRS."
                )

            geometries = []

            for geometry_data, value in shapes(
                mask,
                mask=(mask == 1),
                transform=transform,
            ):
                if int(value) == 1:
                    geometries.append(
                        shape(geometry_data)
                    )

        if not geometries:
            return None, 0, crs

        geometry = unary_union(geometries)

        return geometry, len(geometries), crs

    def execute(
        self,
        source_evidence: Evidence,
    ) -> Evidence:
        result = source_evidence.result

        if not isinstance(result, dict):
            raise ValueError(
                "Temporal change Evidence result must be a dictionary."
            )

        mask = result.get("mask")

        if mask is None:
            raise ValueError(
                "Temporal change Evidence does not expose its mask."
            )

        mask = np.asarray(mask)

        if mask.ndim != 2:
            raise ValueError(
                f"Temporal change mask must be 2D, got {mask.shape}."
            )

        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError(
                "Temporal change mask must contain only 0/1 values."
            )

        transform = result.get("transform")
        crs = result.get("crs")

        if transform is None:
            raise ValueError(
                "Temporal change Evidence is missing transform."
            )

        if crs is None:
            raise ValueError(
                "Temporal change Evidence is missing CRS."
            )

        artifact_id = self._make_artifact_id(
            source_evidence
        )

        mask_path = (
            self.output_dir
            / f"{artifact_id}_mask.tif"
        )

        self._write_mask(
            mask,
            transform,
            crs,
            mask_path,
        )

        geometry, raw_polygon_count, raster_crs = (
            self._polygonize_mask(mask_path)
        )

        if geometry is None:
            raise ValueError(
                "Change mask contains no changed pixels."
            )

        changed_pixels = int(
            np.count_nonzero(mask)
        )
        total_pixels = int(mask.size)

        return Evidence(
            evidence_id=artifact_id,
            source=self.SOURCE,
            task=self.TASK,
            model=self.MODEL,
            sensor=source_evidence.sensor,
            modality=source_evidence.modality,
            timestamp=source_evidence.timestamp,
            geometry={
                "type": "Feature",
                "geometry": geometry.__geo_interface__,
                "properties": {},
            },
            measurement={
                "changed_pixels": changed_pixels,
                "total_pixels": total_pixels,
                "change_percentage": (
                    changed_pixels
                    / max(total_pixels, 1)
                    * 100.0
                ),
                "raw_polygon_count": raw_polygon_count,
            },
            result={
                "mask_path": str(mask_path.resolve()),
                "crs": str(raster_crs),
                "changed_pixels": changed_pixels,
                "total_pixels": total_pixels,
                "raw_polygon_count": raw_polygon_count,
                "geometry_valid": bool(geometry.is_valid),
                "geometry_empty": bool(geometry.is_empty),
            },
            confidence=source_evidence.confidence,
            provenance={
                "source_evidence_id": source_evidence.evidence_id,
                "method": "deterministic_mask_polygonization",
                "mask_path": str(mask_path.resolve()),
                "crs": str(raster_crs),
            },
            metadata={
                "source_model": source_evidence.model,
                "source_task": source_evidence.task,
                "source_confidence": source_evidence.confidence,
            },
        )
