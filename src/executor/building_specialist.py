from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import MultiPolygon, mapping

from src.executor.specialist import Specialist
from src.schemas.evidence import Evidence
from src.geospatial.grounding import pixel_bbox_to_geographic_polygon
from src.training.building_model_inference import load_building_model
from src.training.building_raster_inference import (
    polygonize_building_mask,
    predict_building_raster,
)


class BuildingDetectionSpecialist(Specialist):
    """
    Building detection specialist.

    Normal inference path:
        GeoTIFF
            ->
        trained BuildingUNet
            ->
        building probability/mask
            ->
        geographic building polygons
            ->
        Evidence

    A legacy injected-box path is retained only for regression tests.
    """

    CAPABILITY = "building_detection"

    DEFAULT_CHECKPOINT = (
        "outputs/checkpoints/building_unet_10epoch_dev.pt"
    )

    def __init__(
        self,
        checkpoint_path: str | None = None,
    ) -> None:
        self.checkpoint_path = (
            checkpoint_path
            or self.DEFAULT_CHECKPOINT
        )

        self.model_name = "BuildingUNet_SpaceNet4_dev"

        self._model = None
        self._model_device = None

    @property
    def capability(self) -> str:
        return self.CAPABILITY

    def _get_model(self, device: Any = None):
        """Lazy-load the trained BuildingUNet."""

        import torch

        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if (
            self._model is None
            or self._model_device != device
        ):
            self._model = load_building_model(
                self.checkpoint_path,
                device=device,
            )
            self._model_device = device

        return self._model

    @staticmethod
    def _polygon_confidence(
        polygon: Any,
        probability: np.ndarray,
        transform: Any,
    ) -> float:
        """Calculate mean model probability inside one predicted polygon."""

        polygon_mask = rasterize(
            [(mapping(polygon), 1)],
            out_shape=probability.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        values = probability[
            polygon_mask == 1
        ]

        if values.size == 0:
            return 0.0

        return float(values.mean())

    def _infer_with_model(
        self,
        image_path: Path,
        parameters: dict[str, Any],
        confidence_threshold: float,
        min_area_m2: float,
    ) -> Evidence:
        import torch

        start_time = time.perf_counter()

        device_name = parameters.get(
            "device",
            None,
        )

        if device_name is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            device = torch.device(
                str(device_name)
            )

        model = self._get_model(device)

        raster_result = predict_building_raster(
            str(image_path),
            model,
            device=device,
            threshold=confidence_threshold,
        )

        mask = raster_result["mask"]
        probability = raster_result["probability"]
        transform = raster_result["transform"]
        crs = raster_result["crs"]

        polygon_records = polygonize_building_mask(
            mask,
            transform,
            min_area_m2=min_area_m2,
        )

        polygons = []
        scores = []

        for record in polygon_records:
            polygon = record["geometry"]

            score = self._polygon_confidence(
                polygon,
                probability,
                transform,
            )

            polygons.append(polygon)
            scores.append(score)

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

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
            f"{hash(tuple(round(s, 6) for s in scores)) & 0xFFFFFFFF:08x}"
        )

        return Evidence(
            evidence_id=evidence_id,
            source="BuildingDetectionSpecialist",
            task="building_detection",
            model=self.model_name,
            sensor=parameters.get(
                "sensor",
                "unknown",
            ),
            modality="optical",
            geometry=geometry,
            measurement={
                "building_count": len(polygons),
                "mean_detection_confidence": mean_confidence,
                "min_area_m2": min_area_m2,
                "confidence_threshold": confidence_threshold,
                "predicted_building_pixels": int(
                    mask.sum()
                ),
                "latency_ms": latency_ms,
            },
            result={
                "building_count": len(polygons),
                "has_detections": bool(polygons),
                "crs": crs.to_string(),
                "detection_source": "building_unet",
                "prediction_shape": list(mask.shape),
                "covered_width": raster_result[
                    "covered_width"
                ],
                "covered_height": raster_result[
                    "covered_height"
                ],
            },
            confidence=mean_confidence,
            provenance={
                "image_path": str(image_path),
                "model_name": self.model_name,
                "checkpoint_path": str(
                    Path(self.checkpoint_path).resolve()
                ),
                "confidence_threshold": confidence_threshold,
                "min_area_m2": min_area_m2,
                "device": str(device),
                "perception_status": "CONNECTED",
                "training_dataset": "SpaceNet4",
                "training_scope": (
                    "single-scene development split"
                ),
                "confidence_calibration": "NOT_CALIBRATED",
            },
            metadata={
                "phase": "building_model_connected",
                "warning": (
                    "Development checkpoint trained on one "
                    "SpaceNet scene; not a final benchmark model."
                ),
            },
        )

    def _infer_with_injected_boxes(
        self,
        image_path: Path,
        parameters: dict[str, Any],
        confidence_threshold: float,
    ) -> Evidence:
        """Legacy regression-test path using caller-supplied pixel boxes."""

        start_time = time.perf_counter()

        detected_boxes = parameters.get(
            "detected_boxes",
            [],
        )

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

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

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
            f"BLDG_LEGACY_{image_path.stem}_"
            f"{len(polygons)}_"
            f"{hash(tuple(scores)) & 0xFFFFFFFF:08x}"
        )

        return Evidence(
            evidence_id=evidence_id,
            source="BuildingDetectionSpecialist",
            task="building_detection",
            model="building_grounding_baseline",
            sensor=parameters.get(
                "sensor",
                "unknown",
            ),
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
                "detection_source": "injected_pixel_boxes",
            },
            confidence=mean_confidence,
            provenance={
                "image_path": str(image_path),
                "model_name": "building_grounding_baseline",
                "confidence_threshold": confidence_threshold,
                "perception_status": "LEGACY_TEST_MODE",
            },
            metadata={
                "phase": "building_grounding_baseline",
                "warning": (
                    "Injected boxes are test inputs, not model detections."
                ),
            },
        )

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

        if isinstance(
            confidence_threshold,
            bool,
        ):
            raise ValueError(
                "confidence_threshold must be numeric."
            )

        try:
            confidence_threshold = float(
                confidence_threshold
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "confidence_threshold must be numeric."
            ) from exc

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0.0 and 1.0."
            )

        min_area_m2 = parameters.get(
            "min_area_m2",
            4.0,
        )

        try:
            min_area_m2 = float(min_area_m2)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "min_area_m2 must be numeric."
            ) from exc

        if min_area_m2 < 0.0:
            raise ValueError(
                "min_area_m2 must be non-negative."
            )

        # Keep explicit regression-test compatibility.
        if "detected_boxes" in parameters:
            return self._infer_with_injected_boxes(
                image_path,
                parameters,
                confidence_threshold,
            )

        # Normal production/development path.
        return self._infer_with_model(
            image_path,
            parameters,
            confidence_threshold,
            min_area_m2,
        )
