from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from src.schemas.evidence import Evidence
from src.executor.specialist import Specialist


class ChangeSpecialist(Specialist):
    """
    Deterministic bi-temporal raster change detector.

    This is the baseline temporal specialist for SATQuery.
    It compares two co-registered raster images and produces
    standardized Evidence.

    It does not use an LLM and does not claim semantic
    understanding of the detected changes.
    """

    CAPABILITY = "temporal_analysis"

    def __init__(
        self,
        default_threshold: float = 0.15,
    ) -> None:
        if not 0.0 < default_threshold <= 1.0:
            raise ValueError(
                "default_threshold must be greater than 0 and at most 1"
            )

        self.default_threshold = default_threshold

    @property
    def capability(self) -> str:
        return self.CAPABILITY

    def _validate_inputs(
        self,
        inputs: list[str],
    ) -> tuple[Path, Path]:
        if len(inputs) != 2:
            raise ValueError(
                "ChangeSpecialist requires exactly two raster inputs."
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

        return before, after

    @staticmethod
    def _read_comparable_data(
        path: Path,
    ) -> tuple[np.ndarray, Any, str | None, tuple[int, int]]:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError(
                    f"Raster has no CRS: {path}"
                )

            data = dataset.read()

            if data.size == 0:
                raise ValueError(
                    f"Raster contains no pixel data: {path}"
                )

            # Convert each band independently to float32.
            data = data.astype(np.float32)

            return (
                data,
                dataset.transform,
                dataset.crs.to_string(),
                (dataset.height, dataset.width),
            )

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """
        Normalize each band using its own min/max range.

        Constant bands become zero arrays.
        """

        normalized = np.zeros_like(data, dtype=np.float32)

        for band_index in range(data.shape[0]):
            band = data[band_index]

            finite = np.isfinite(band)

            if not finite.any():
                continue

            valid = band[finite]
            minimum = float(valid.min())
            maximum = float(valid.max())

            if maximum <= minimum:
                continue

            normalized[band_index] = np.clip(
                (band - minimum) / (maximum - minimum),
                0.0,
                1.0,
            )

        return normalized

    @staticmethod
    def _make_evidence_id(
        before: Path,
        after: Path,
    ) -> str:
        return (
            f"CHANGE_{before.stem}_TO_{after.stem}"
        )

    def infer(
        self,
        inputs: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> Evidence:
        parameters = parameters or {}

        before, after = self._validate_inputs(inputs)

        threshold = parameters.get(
            "threshold",
            self.default_threshold,
        )

        if isinstance(threshold, bool) or not isinstance(
            threshold,
            (int, float),
        ):
            raise ValueError(
                "threshold must be a numeric value."
            )

        threshold = float(threshold)

        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                "threshold must be greater than 0 and at most 1"
            )

        before_data, before_transform, before_crs, before_shape = (
            self._read_comparable_data(before)
        )

        after_data, after_transform, after_crs, after_shape = (
            self._read_comparable_data(after)
        )

        if before_shape != after_shape:
            raise ValueError(
                "Before and after rasters must have identical dimensions."
            )

        if before_data.shape != after_data.shape:
            raise ValueError(
                "Before and after rasters must have the same band count."
            )

        if before_crs != after_crs:
            raise ValueError(
                "Before and after rasters must use the same CRS."
            )

        if before_transform != after_transform:
            raise ValueError(
                "Before and after rasters must have matching transforms."
            )

        before_normalized = self._normalize(before_data)
        after_normalized = self._normalize(after_data)

        difference = np.abs(
            after_normalized - before_normalized
        )

        # Aggregate multispectral differences across bands.
        difference_score = np.mean(
            difference,
            axis=0,
        )

        change_mask = difference_score >= threshold

        total_pixels = change_mask.size

        if total_pixels == 0:
            raise ValueError(
                "Raster contains zero comparable pixels."
            )

        changed_pixels = int(change_mask.sum())

        change_percentage = (
            changed_pixels / total_pixels
        ) * 100.0

        mean_change = float(
            np.mean(difference_score)
        )

        max_change = float(
            np.max(difference_score)
        )

        # Baseline confidence reflects signal strength,
        # not calibrated model probability.
        confidence = float(
            min(
                1.0,
                max(
                    0.0,
                    mean_change / max(threshold, 1e-6),
                ),
            )
        )

        result = {
            "changed": changed_pixels > 0,
            "change_percentage": change_percentage,
            "changed_pixels": changed_pixels,
            "total_pixels": int(total_pixels),
            "mean_change": mean_change,
            "max_change": max_change,
            "threshold": threshold,
        }

        provenance = {
            "method": "normalized_absolute_raster_difference",
            "specialist": self.__class__.__name__,
            "confidence_note": (
                "Confidence is a deterministic signal-strength "
                "score and is not a calibrated probability."
            ),
            "before": str(before.resolve()),
            "after": str(after.resolve()),
            "crs": before_crs,
        }

        return Evidence(
            evidence_id=self._make_evidence_id(
                before,
                after,
            ),
            source="change_specialist",
            task=self.capability,
            model="deterministic_change_baseline",
            sensor=None,
            modality="optical",
            geometry=None,
            measurement={
                "change_percentage": change_percentage,
                "mean_change": mean_change,
                "max_change": max_change,
            },
            result=result,
            confidence=confidence,
            provenance=provenance,
            metadata={
                "threshold": threshold,
                "shape": before_shape,
                "band_count": int(before_data.shape[0]),
            },
        )
