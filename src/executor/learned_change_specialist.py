from __future__ import annotations

import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.executor.specialist import Specialist
from src.schemas.evidence import Evidence
from src.training.change_model_inference import (
    load_change_model,
)
from src.training.change_raster_inference import (
    predict_change_raster,
)


class LearnedChangeSpecialist(Specialist):
    """
    Learned bi-temporal optical change specialist.

    Uses the trained ChangeUNet checkpoint and produces
    evidence compatible with the existing SATQuery evidence layer.

    This checkpoint is development-only and trained on
    controlled synthetic change data.
    """

    DEFAULT_CHECKPOINT = (
        "outputs/checkpoints/"
        "change_unet_synthetic_dev.pt"
    )

    def __init__(
        self,
        checkpoint_path: str | None = None,
    ) -> None:

        self.checkpoint_path = (
            checkpoint_path
            or self.DEFAULT_CHECKPOINT
        )

        self._model = None
        self._device = None

    @property
    def capability(self) -> str:
        return "temporal_analysis"

    def _get_model(self):
        if self._model is None:
            self._model, self._device = (
                load_change_model(
                    self.checkpoint_path
                )
            )

        return self._model, self._device

    def unload(self) -> None:
        """Release the learned model after its Evidence has been recorded."""
        self._model = None
        self._device = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            # CPU-only environments do not need CUDA cache cleanup.
            pass

    @staticmethod
    def _make_evidence_id(
        before: Path,
        after: Path,
    ) -> str:

        payload = (
            f"{before.resolve()}|"
            f"{after.resolve()}|"
            "change_unet"
        )

        digest = hashlib.sha256(
            payload.encode()
        ).hexdigest()[:12]

        return (
            f"CHG_{before.stem}_"
            f"{after.stem}_"
            f"{digest}"
        )

    def infer(
        self,
        inputs: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> Evidence:

        if len(inputs) != 2:
            raise ValueError(
                "LearnedChangeSpecialist requires "
                "exactly two raster inputs."
            )

        parameters = parameters or {}

        # Optional real-data temporal metadata.
        #
        # These values describe the acquisition pair and do not
        # alter ChangeUNet inference itself.
        before_timestamp = parameters.get(
            "before_timestamp"
        )

        after_timestamp = parameters.get(
            "after_timestamp"
        )

        sensor = parameters.get(
            "sensor",
            "Sentinel-2"
        )

        temporal_separation_seconds = None
        temporal_separation_days = None

        if before_timestamp and after_timestamp:

            before_dt = datetime.fromisoformat(
                str(before_timestamp).replace(
                    "Z",
                    "+00:00"
                )
            )

            after_dt = datetime.fromisoformat(
                str(after_timestamp).replace(
                    "Z",
                    "+00:00"
                )
            )

            if before_dt.tzinfo is None:
                before_dt = before_dt.replace(
                    tzinfo=timezone.utc
                )

            if after_dt.tzinfo is None:
                after_dt = after_dt.replace(
                    tzinfo=timezone.utc
                )

            temporal_separation_seconds = (
                after_dt - before_dt
            ).total_seconds()

            if temporal_separation_seconds < 0:
                raise ValueError(
                    "after_timestamp must not precede "
                    "before_timestamp."
                )

            temporal_separation_days = (
                temporal_separation_seconds
                / 86400.0
            )

        threshold = float(
            parameters.get(
                "change_threshold",
                0.5,
            )
        )

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "change_threshold must be between 0 and 1."
            )

        before = Path(inputs[0])
        after = Path(inputs[1])

        if not before.exists():
            raise FileNotFoundError(
                f"Before raster does not exist: {before}"
            )

        if not after.exists():
            raise FileNotFoundError(
                f"After raster does not exist: {after}"
            )

        model, device = self._get_model()

        prediction = predict_change_raster(
            model,
            str(before),
            str(after),
            device=device,
            threshold=threshold,
        )

        probability = prediction["probability"]
        mask = prediction["mask"]

        changed_pixels = int(mask.sum())
        total_pixels = int(mask.size)

        change_percentage = (
            changed_pixels
            / max(total_pixels, 1)
            * 100.0
        )

        mean_probability = float(
            probability.mean()
        )

        max_probability = float(
            probability.max()
        )

        confidence = float(
            np.mean(
                np.maximum(
                    probability,
                    1.0 - probability,
                )
            )
        )

        result = {
            "changed": changed_pixels > 0,
            "change_percentage": change_percentage,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "mean_probability": mean_probability,
            "max_probability": max_probability,
            "threshold": threshold,
            "prediction_shape": list(mask.shape),
            "covered_width": prediction["covered_width"],
            "covered_height": prediction["covered_height"],
            # Raster evidence contract consumed by ChangeGeospatializer.
            "mask": mask,
            "transform": prediction["transform"],
            "crs": prediction["crs"],
        }

        provenance = {
            "method": "siamese_change_unet",
            "specialist": self.__class__.__name__,
            "checkpoint": str(
                Path(
                    self.checkpoint_path
                ).resolve()
            ),
            "before": str(
                before.resolve()
            ),
            "after": str(
                after.resolve()
            ),
            "image_path": str(after.resolve()),
            "crs": str(
                prediction["crs"]
            ),
            "sensor": sensor,
            "before_timestamp": before_timestamp,
            "after_timestamp": after_timestamp,
            "temporal_separation_seconds":
                temporal_separation_seconds,
            "temporal_separation_days":
                temporal_separation_days,
            "perception_status": "CONNECTED",
            "training_dataset": (
                "SpaceNet4-derived synthetic "
                "development data"
            ),
            "training_scope": (
                "single-scene development split"
            ),
            "confidence_calibration": (
                "NOT_CALIBRATED"
            ),
            "development_warning": (
                "This checkpoint was trained on "
                "controlled synthetic changes and "
                "is not a real-world benchmark model."
            ),
        }

        return Evidence(
            evidence_id=self._make_evidence_id(
                before,
                after,
            ),
            source="learned_change_specialist",
            task=self.capability,
            model="ChangeUNet_SyntheticDev",
            sensor=sensor,
            modality="optical",
            # Temporal change refers to the after image, allowing the
            # existing alignment checker to compare it with SAR evidence.
            timestamp=after_timestamp,
            geometry=None,
            measurement={
                "change_percentage": (
                    change_percentage
                ),
                "changed_pixels": (
                    changed_pixels
                ),
                "total_pixels": total_pixels,
                "mean_probability": (
                    mean_probability
                ),
                "max_probability": (
                    max_probability
                ),
            },
            result=result,
            confidence=confidence,
            provenance=provenance,
            metadata={
                "threshold": threshold,
                "shape": list(mask.shape),
                "band_count": 4,
                "device": str(device),
                "sensor": sensor,
                "before_timestamp": before_timestamp,
                "after_timestamp": after_timestamp,
                "temporal_separation_seconds":
                    temporal_separation_seconds,
                "temporal_separation_days":
                    temporal_separation_days,
            },
        )
