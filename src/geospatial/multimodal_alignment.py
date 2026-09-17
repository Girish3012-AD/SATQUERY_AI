from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import rasterio
from rasterio.warp import transform_bounds

from src.schemas.evidence import Evidence


def _parse_timestamp(timestamp: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into a datetime."""
    if not timestamp:
        return None

    try:
        value = timestamp.strip()

        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _bounds_in_common_crs(
    bounds,
    source_crs,
    target_crs,
):
    """Transform raster bounds into a common CRS."""
    if source_crs is None or target_crs is None:
        return None

    if source_crs == target_crs:
        return tuple(bounds)

    return tuple(
        transform_bounds(
            source_crs,
            target_crs,
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )
    )


def _bounds_overlap(
    first_bounds,
    second_bounds,
) -> tuple[bool, float]:
    """
    Determine whether two bounding boxes overlap and estimate
    overlap fraction relative to the smaller bounding box.
    """
    left = max(first_bounds[0], second_bounds[0])
    bottom = max(first_bounds[1], second_bounds[1])
    right = min(first_bounds[2], second_bounds[2])
    top = min(first_bounds[3], second_bounds[3])

    if right <= left or top <= bottom:
        return False, 0.0

    intersection_area = (
        (right - left)
        * (top - bottom)
    )

    first_area = max(
        (first_bounds[2] - first_bounds[0])
        * (first_bounds[3] - first_bounds[1]),
        0.0,
    )

    second_area = max(
        (second_bounds[2] - second_bounds[0])
        * (second_bounds[3] - second_bounds[1]),
        0.0,
    )

    smaller_area = min(
        first_area,
        second_area,
    )

    if smaller_area <= 0:
        return True, 0.0

    return (
        True,
        intersection_area / smaller_area,
    )


def _raster_metadata(path: str) -> dict[str, Any]:
    """Read the geospatial metadata required for compatibility checks."""
    raster = Path(path)

    if not raster.exists():
        raise FileNotFoundError(
            f"Raster does not exist: {raster}"
        )

    with rasterio.open(raster) as dataset:
        return {
            "crs": dataset.crs,
            "bounds": dataset.bounds,
            "resolution": tuple(dataset.res),
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
        }


def validate_optical_sar_alignment(
    optical_path: str,
    sar_path: str,
) -> dict:
    """
    Validate strict raster-grid alignment.

    This legacy function is intentionally preserved for callers
    that require identical computational grids.
    """
    optical = Path(optical_path)
    sar = Path(sar_path)

    if not optical.exists():
        raise FileNotFoundError(
            f"Optical raster does not exist: {optical}"
        )

    if not sar.exists():
        raise FileNotFoundError(
            f"SAR raster does not exist: {sar}"
        )

    with rasterio.open(optical) as optical_ds:
        optical_crs = optical_ds.crs
        optical_transform = optical_ds.transform
        optical_width = optical_ds.width
        optical_height = optical_ds.height
        optical_bounds = optical_ds.bounds
        optical_res = optical_ds.res
        optical_count = optical_ds.count

    with rasterio.open(sar) as sar_ds:
        sar_crs = sar_ds.crs
        sar_transform = sar_ds.transform
        sar_width = sar_ds.width
        sar_height = sar_ds.height
        sar_bounds = sar_ds.bounds
        sar_res = sar_ds.res
        sar_count = sar_ds.count

    same_crs = optical_crs == sar_crs

    same_shape = (
        optical_width == sar_width
        and optical_height == sar_height
    )

    same_transform = (
        optical_transform == sar_transform
    )

    same_bounds = (
        optical_bounds == sar_bounds
    )

    same_resolution = (
        optical_res == sar_res
    )

    aligned = all(
        [
            same_crs,
            same_shape,
            same_transform,
            same_bounds,
            same_resolution,
        ]
    )

    return {
        "aligned": aligned,
        "same_crs": same_crs,
        "same_shape": same_shape,
        "same_transform": same_transform,
        "same_bounds": same_bounds,
        "same_resolution": same_resolution,
        "optical_bands": optical_count,
        "sar_bands": sar_count,
        "optical_crs": str(optical_crs),
        "sar_crs": str(sar_crs),
    }


def validate_optical_sar_evidence_compatibility(
    optical_evidence: Evidence,
    sar_evidence: Evidence,
    *,
    max_temporal_separation_seconds: float = 86400.0,
    max_resolution_ratio: float = 4.0,
) -> dict[str, Any]:
    """
    Validate whether optical and SAR Evidence can be combined.

    This is an evidence-level compatibility check rather than a
    strict raster-grid alignment check.

    The result deliberately distinguishes:
        compatible
        incompatible
        uncertain

    because missing metadata should not silently become a positive
    compatibility decision.
    """
    if optical_evidence.modality != "optical":
        return {
            "status": "incompatible",
            "reason": (
                "First evidence item must have "
                "modality='optical'."
            ),
        }

    if sar_evidence.modality != "sar":
        return {
            "status": "incompatible",
            "reason": (
                "Second evidence item must have "
                "modality='sar'."
            ),
        }

    optical_path = (
        optical_evidence.provenance.get("image_path")
        or (
            optical_evidence.result.get("image_path")
            if isinstance(
                optical_evidence.result,
                dict,
            )
            else None
        )
    )

    sar_path = (
        sar_evidence.provenance.get("image_path")
        or (
            sar_evidence.result.get("image_path")
            if isinstance(
                sar_evidence.result,
                dict,
            )
            else None
        )
    )

    if not optical_path or not sar_path:
        return {
            "status": "uncertain",
            "reason": (
                "Raster paths are missing from evidence "
                "provenance/result."
            ),
        }

    optical = _raster_metadata(str(optical_path))
    sar = _raster_metadata(str(sar_path))

    spatial_reasons: list[str] = []

    if optical["crs"] is None or sar["crs"] is None:
        crs_compatible = False
        reprojection_possible = False
        spatial_reasons.append(
            "One or both rasters have no CRS."
        )
        overlap = False
        overlap_fraction = 0.0
    else:
        crs_compatible = (
            optical["crs"] == sar["crs"]
        )

        reprojection_possible = True

        common_crs = optical["crs"]

        sar_bounds_common = _bounds_in_common_crs(
            sar["bounds"],
            sar["crs"],
            common_crs,
        )

        optical_bounds_common = _bounds_in_common_crs(
            optical["bounds"],
            optical["crs"],
            common_crs,
        )

        overlap, overlap_fraction = _bounds_overlap(
            optical_bounds_common,
            sar_bounds_common,
        )

        if not crs_compatible:
            spatial_reasons.append(
                "Optical and SAR rasters use different CRSs."
            )

        if not overlap:
            spatial_reasons.append(
                "Optical and SAR footprints do not overlap."
            )

    optical_resolution = max(
        abs(float(optical["resolution"][0])),
        abs(float(optical["resolution"][1])),
    )

    sar_resolution = max(
        abs(float(sar["resolution"][0])),
        abs(float(sar["resolution"][1])),
    )

    if optical_resolution <= 0 or sar_resolution <= 0:
        resolution_compatible = False
        resolution_ratio = None
    else:
        resolution_ratio = max(
            optical_resolution / sar_resolution,
            sar_resolution / optical_resolution,
        )

        resolution_compatible = (
            resolution_ratio <= max_resolution_ratio
        )

        if not resolution_compatible:
            spatial_reasons.append(
                "Native resolutions differ beyond the "
                "configured compatibility ratio."
            )

    optical_timestamp = _parse_timestamp(
        optical_evidence.timestamp
    )

    sar_timestamp = _parse_timestamp(
        sar_evidence.timestamp
    )

    temporal_reasons: list[str] = []

    if (
        optical_timestamp is None
        or sar_timestamp is None
    ):
        temporal_compatible = None
        timestamp_delta_seconds = None
        temporal_reasons.append(
            "One or both acquisition timestamps are missing "
            "or invalid."
        )
    else:
        timestamp_delta_seconds = abs(
            (
                optical_timestamp
                - sar_timestamp
            ).total_seconds()
        )

        temporal_compatible = (
            timestamp_delta_seconds
            <= max_temporal_separation_seconds
        )

        if not temporal_compatible:
            temporal_reasons.append(
                "Acquisition times exceed the configured "
                "temporal compatibility window."
            )

    modality_compatible = (
        optical_evidence.modality == "optical"
        and sar_evidence.modality == "sar"
    )

    sensor_compatible = (
        optical_evidence.sensor is not None
        and sar_evidence.sensor is not None
    )

    provenance_complete = bool(
        optical_evidence.provenance
        and sar_evidence.provenance
    )

    reasons: list[str] = []

    reasons.extend(spatial_reasons)
    reasons.extend(temporal_reasons)

    if not modality_compatible:
        reasons.append(
            "Evidence modalities are not optical + SAR."
        )

    if not sensor_compatible:
        reasons.append(
            "One or both sensor identities are missing."
        )

    if not provenance_complete:
        reasons.append(
            "One or both evidence items lack provenance."
        )

    hard_incompatibility = (
        not crs_compatible
        or not overlap
        or not resolution_compatible
        or not modality_compatible
    )

    temporal_incompatibility = (
        temporal_compatible is False
    )

    if hard_incompatibility or temporal_incompatibility:
        status = "incompatible"
    elif (
        temporal_compatible is None
        or not sensor_compatible
        or not provenance_complete
    ):
        status = "uncertain"
    else:
        status = "compatible"

    if not reasons:
        reasons.append(
            "Optical and SAR evidence satisfy the configured "
            "compatibility checks."
        )

    return {
        "status": status,
        "evidence_ids": [
            optical_evidence.evidence_id,
            sar_evidence.evidence_id,
        ],
        "modalities": [
            optical_evidence.modality,
            sar_evidence.modality,
        ],
        "sensors": [
            optical_evidence.sensor,
            sar_evidence.sensor,
        ],
        "spatial": {
            "crs_compatible": crs_compatible,
            "reprojection_possible": reprojection_possible,
            "overlap": overlap,
            "overlap_fraction": overlap_fraction,
            "optical_crs": str(optical["crs"]),
            "sar_crs": str(sar["crs"]),
            "optical_bounds": tuple(
                optical["bounds"]
            ),
            "sar_bounds": tuple(
                sar["bounds"]
            ),
        },
        "resolution": {
            "optical": optical_resolution,
            "sar": sar_resolution,
            "ratio": resolution_ratio,
            "compatible": resolution_compatible,
        },
        "temporal": {
            "optical_timestamp": (
                optical_evidence.timestamp
            ),
            "sar_timestamp": (
                sar_evidence.timestamp
            ),
            "timestamp_delta_seconds": (
                timestamp_delta_seconds
            ),
            "max_allowed_seconds": (
                max_temporal_separation_seconds
            ),
            "compatible": temporal_compatible,
        },
        "sensor": {
            "optical": optical_evidence.sensor,
            "sar": sar_evidence.sensor,
        },
        "provenance": {
            "complete": provenance_complete,
            "optical": optical_evidence.provenance,
            "sar": sar_evidence.provenance,
        },
        "reasons": reasons,
    }
