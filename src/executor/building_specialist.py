from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import rasterio
from shapely.geometry import MultiPolygon, mapping

from src.executor.specialist import Specialist
from src.schemas.evidence import Evidence
from src.geospatial.grounding import pixel_bbox_to_geographic_polygon


class BuildingDetectionSpecialist(Specialist):
    """
    Building detection specialist.

    This phase provides the geospatial/evidence bridge:
        pixel bounding boxes
            ->
        CRS-aware geographic polygons
            ->
        Evidence

    Actual visual building perception is intentionally kept separate and
    will be connected after this contract is validated.
    """

    CAPABILITY = "building_detection"

    def __init__(self) -> None:
        self.model_name = "building_grounding_baseline"

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
                "BuildingDetectionSpecialist requires at least one raster input."
            )

        image_path = Path(inputs[0])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Input raster does not exist: {image_path}"
            )

        parameters = parameters or {}

        confidence_threshold = parameters.get(
            "confidence_threshold",
            0.5,
        )

        if isinstance(confidence_threshold, bool):
            raise ValueError(
                "confidence_threshold must be numeric."
            )

        try:
            confidence_threshold = float(confidence_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "confidence_threshold must be numeric."
            ) from exc

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0.0 and 1.0."
            )

        # These boxes are deliberately supplied by the caller.
        #
        # This is NOT visual detection.
        # It allows us to test the complete geospatial/evidence path
        # before connecting an actual perception model.
        detected_boxes = parameters.get("detected_boxes", [])

        start_time = time.perf_counter()

        with rasterio.open(image_path) as dataset:
            if dataset.crs is None:
                raise ValueError(
                    "Building detection requires a raster with a valid CRS."
                )

            transform = dataset.transform
            crs = dataset.crs.to_string()

            width = dataset.width
            height = dataset.height

            polygons = []
            scores = []

            for detection in detected_boxes:
                if len(detection) != 5:
                    raise ValueError(
                        "Each detection must contain "
                        "(x_min, y_min, x_max, y_max, confidence)."
                    )

                x_min, y_min, x_max, y_max, score = detection

                score = float(score)

                if not 0.0 <= score <= 1.0:
                    raise ValueError(
                        "Detection confidence must be between 0.0 and 1.0."
                    )

                # Reject boxes outside the raster.
                if (
                    x_min < 0
                    or y_min < 0
                    or x_max > width
                    or y_max > height
                ):
                    raise ValueError(
                        "Detection bounding box lies outside the raster."
                    )

                if score < confidence_threshold:
                    continue

                polygon = pixel_bbox_to_geographic_polygon(
                    (
                        float(x_min),
                        float(y_min),
                        float(x_max),
                        float(y_max),
                    ),
                    transform,
                )

                polygons.append(polygon)
                scores.append(score)

        latency_ms = (time.perf_counter() - start_time) * 1000

        geometry = (
            mapping(MultiPolygon(polygons))
            if polygons
            else None
        )

        mean_confidence = (
            float(sum(scores) / len(scores))
            if scores
            else 0.0
        )

        evidence_id = (
            f"BLDG_{image_path.stem}_"
            f"{len(polygons)}_"
            f"{hash(tuple(scores)) & 0xFFFFFFFF:08x}"
        )

        return Evidence(
            evidence_id=evidence_id,
            source="BuildingDetectionSpecialist",
            task="building_detection",
            model=self.model_name,
            sensor=parameters.get("sensor", "unknown"),
            modality="optical",
            geometry=geometry,
            measurement={
                "building_count": len(polygons),
                "mean_detection_confidence": mean_confidence,
                "latency_ms": latency_ms,
            },
            result={
                "building_count": len(polygons),
                "has_detections": bool(polygons),
                "crs": crs,
                "detection_source": (
                    "injected_pixel_boxes"
                    if detected_boxes
                    else "none"
                ),
            },
            confidence=mean_confidence,
            provenance={
                "image_path": str(image_path),
                "model_name": self.model_name,
                "confidence_threshold": confidence_threshold,
                "perception_status": (
                    "NOT_CONNECTED"
                ),
            },
            metadata={
                "phase": "building_grounding_baseline",
                "warning": (
                    "Injected boxes are test inputs, not model detections."
                ),
            },
        )
