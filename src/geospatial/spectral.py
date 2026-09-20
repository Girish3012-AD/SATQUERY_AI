"""
spectral.py — Deterministic spectral index calculations.

No LLM is involved. All operations are NumPy raster computations.
Results are described as "potential" / "candidate" — not confirmed
geophysical classifications — unless a validated model confirms them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpectralResult:
    """Container for a computed spectral index raster."""

    index_name: str
    data: np.ndarray          # float32, NaN where nodata
    threshold: float
    mask: np.ndarray          # bool, True = candidate pixels
    candidate_pixel_count: int
    nodata_pixel_count: int
    transform: Any            # rasterio Affine
    crs: str
    resolution_m: tuple[float, float]
    provenance: dict[str, Any]


def compute_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    *,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Compute NDWI = (Green - NIR) / (Green + NIR).

    Returns float32 array in [-1, 1].
    Pixels where both bands are zero or nodata are set to NaN.
    Division by zero is handled safely.
    """
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)

    if nodata is not None:
        nodata_mask = (green == nodata) | (nir == nodata)
        green[nodata_mask] = np.nan
        nir[nodata_mask] = np.nan

    denominator = green + nir

    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(
            denominator == 0,
            np.nan,
            (green - nir) / denominator,
        ).astype(np.float32)

    return ndwi


def water_candidate_mask(
    ndwi: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Return a boolean mask where NDWI >= threshold.

    Pixels with NaN NDWI are excluded (False).
    Threshold of 0.0 is standard for open water detection.
    """
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(
            "NDWI threshold must be in [-1.0, 1.0]."
        )

    return np.where(
        np.isnan(ndwi),
        False,
        ndwi >= threshold,
    ).astype(bool)


def analyse_sentinel2_ndwi(
    green_band: np.ndarray,
    nir_band: np.ndarray,
    *,
    transform: Any,
    crs: str,
    resolution_m: tuple[float, float],
    nodata: float | None = None,
    threshold: float = 0.0,
    scene_id: str = "unknown",
    acquisition_date: str = "unknown",
    bands_used: tuple[str, str] = ("B03", "B08"),
) -> SpectralResult:
    """
    Full NDWI analysis for a Sentinel-2 scene.

    Returns a SpectralResult with the index raster, candidate mask,
    and full provenance.  The label "potential_water_candidate" is
    used deliberately — not "flood" or "confirmed water".
    """
    ndwi = compute_ndwi(green_band, nir_band, nodata=nodata)
    mask = water_candidate_mask(ndwi, threshold=threshold)

    candidate_px = int(np.sum(mask))
    nodata_px = int(np.sum(np.isnan(ndwi)))

    provenance: dict[str, Any] = {
        "scene_id": scene_id,
        "acquisition_date": acquisition_date,
        "sensor": "Sentinel-2",
        "bands": list(bands_used),
        "index": "NDWI",
        "formula": "NDWI = (B03 - B08) / (B03 + B08)",
        "threshold": threshold,
        "label": "potential_water_candidate",
        "note": (
            "NDWI >= threshold indicates spectral similarity to water. "
            "This is NOT confirmed flooding without additional validation."
        ),
    }

    return SpectralResult(
        index_name="NDWI",
        data=ndwi,
        threshold=threshold,
        mask=mask,
        candidate_pixel_count=candidate_px,
        nodata_pixel_count=nodata_px,
        transform=transform,
        crs=crs,
        resolution_m=resolution_m,
        provenance=provenance,
    )
