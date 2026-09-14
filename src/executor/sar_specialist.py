from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from src.schemas import Evidence

from .specialist import Specialist


class SARSpecialist(Specialist):
    """
    Deterministic SAR characterization specialist.

    This component validates and summarizes a SAR raster and returns
    normalized SATQuery Evidence.

    It is intentionally not presented as a learned SAR classifier or
    RISAT detector. Its purpose is to establish the real SAR evidence
    contract and provide deterministic SAR measurements that can later
    support learned SAR perception.
    """

    CAPABILITY = "sar_analysis"
    MODEL_NAME = "SAR_Deterministic_Characterizer"

    @property
    def capability(self) -> str:
        return self.CAPABILITY

    def infer(
        self,
        inputs: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> Evidence:
        if not inputs:
            raise ValueError(
                "SARSpecialist requires at least one raster input."
            )

        image_path = Path(inputs[0])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Input SAR raster does not exist: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Input SAR path is not a file: {image_path}"
            )

        parameters = parameters or {}

        start_time = time.perf_counter()

        with rasterio.open(image_path) as dataset:
            if dataset.count != 1:
                raise ValueError(
                    "SARSpecialist currently requires a single-band SAR raster."
                )

            if dataset.crs is None:
                raise ValueError(
                    "SARSpecialist requires a raster with a valid CRS."
                )

            if dataset.width <= 0 or dataset.height <= 0:
                raise ValueError(
                    "SAR raster must have positive spatial dimensions."
                )

            array = dataset.read(1, masked=True)

            values = np.asarray(array.compressed(), dtype=np.float64)

            if values.size == 0:
                raise ValueError(
                    "SAR raster contains no valid pixels."
                )

            finite = values[np.isfinite(values)]

            if finite.size == 0:
                raise ValueError(
                    "SAR raster contains no finite valid pixels."
                )

            if finite.size != values.size:
                raise ValueError(
                    "SAR raster contains non-finite valid pixel values."
                )

            crs = dataset.crs.to_string()
            transform = dataset.transform

            resolution_x = abs(float(transform.a))
            resolution_y = abs(float(transform.e))

            bounds = (
                float(dataset.bounds.left),
                float(dataset.bounds.bottom),
                float(dataset.bounds.right),
                float(dataset.bounds.top),
            )

            nodata = dataset.nodata

            width = int(dataset.width)
            height = int(dataset.height)

        mean_value = float(np.mean(finite))
        std_value = float(np.std(finite))
        min_value = float(np.min(finite))
        max_value = float(np.max(finite))

        p01, p05, p25, p50, p75, p95, p99 = np.percentile(
            finite,
            [1, 5, 25, 50, 75, 95, 99],
        )

        valid_fraction = float(
            finite.size / (width * height)
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        sensor = str(
            parameters.get(
                "sensor",
                "unknown",
            )
        )

        polarization = parameters.get(
            "polarization"
        )

        band = parameters.get(
            "band"
        )

        # Deterministic characterization is not a calibrated probability.
        # A high-quality input contract receives 0.80 as an operational
        # evidence confidence, while the metadata records that this is not
        # a learned detection confidence.
        confidence = 0.80

        evidence_id = (
            f"SAR_{image_path.stem}_"
            f"{hash((
                round(mean_value, 6),
                round(std_value, 6),
                round(min_value, 6),
                round(max_value, 6),
            )) & 0xFFFFFFFF:08x}"
        )

        measurement: dict[str, Any] = {
            "valid_pixel_count": int(finite.size),
            "valid_fraction": valid_fraction,
            "mean_backscatter_db": mean_value,
            "std_backscatter_db": std_value,
            "min_backscatter_db": min_value,
            "max_backscatter_db": max_value,
            "p01_backscatter_db": float(p01),
            "p05_backscatter_db": float(p05),
            "p25_backscatter_db": float(p25),
            "p50_backscatter_db": float(p50),
            "p75_backscatter_db": float(p75),
            "p95_backscatter_db": float(p95),
            "p99_backscatter_db": float(p99),
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
            "width": width,
            "height": height,
            "latency_ms": latency_ms,
        }

        if nodata is not None:
            measurement["nodata"] = float(nodata)

        if polarization is not None:
            measurement["polarization"] = str(
                polarization
            )

        if band is not None:
            measurement["band"] = str(
                band
            )

        result: dict[str, Any] = {
            "analysis_type": "sar_characterization",
            "has_valid_data": True,
            "crs": crs,
            "width": width,
            "height": height,
            "bounds": bounds,
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
            "valid_pixel_count": int(finite.size),
            "valid_fraction": valid_fraction,
        }

        if polarization is not None:
            result["polarization"] = str(
                polarization
            )

        if band is not None:
            result["band"] = str(
                band
            )

        return Evidence(
            evidence_id=evidence_id,
            source="SARSpecialist",
            task=self.CAPABILITY,
            model=self.MODEL_NAME,
            sensor=sensor,
            modality="sar",
            geometry=None,
            measurement=measurement,
            result=result,
            confidence=confidence,
            provenance={
                "image_path": str(image_path.resolve()),
                "model_name": self.MODEL_NAME,
                "analysis_type": "deterministic_sar_characterization",
                "device": "cpu",
                "perception_status": "CONNECTED",
                "confidence_calibration": "NOT_CALIBRATED",
            },
            metadata={
                "phase": "sar_specialist_connected",
                "deterministic": True,
                "warning": (
                    "This specialist characterizes SAR raster statistics; "
                    "it does not perform learned object or flood detection."
                ),
            },
        )
