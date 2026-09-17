from datetime import datetime, timezone

import rasterio
from rasterio.transform import from_origin

from src.geospatial.multimodal_alignment import (
    validate_optical_sar_evidence_compatibility,
)
from src.schemas.evidence import Evidence


def _create_raster(
    path,
    *,
    crs="EPSG:32643",
    width=100,
    height=100,
    resolution=10.0,
    west=362000.0,
    north=2065000.0,
):
    transform = from_origin(
        west,
        north,
        resolution,
        resolution,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint16",
        crs=crs,
        transform=transform,
    ) as dst:
        import numpy as np

        dst.write(
            np.ones(
                (height, width),
                dtype=np.uint16,
            ),
            1,
        )


def _evidence(
    *,
    evidence_id,
    sensor,
    modality,
    timestamp,
    path,
    crs="EPSG:32643",
    resolution=10.0,
):
    return Evidence(
        evidence_id=evidence_id,
        source="test",
        task="test",
        model="test-model",
        sensor=sensor,
        modality=modality,
        timestamp=timestamp,
        geometry=None,
        measurement={
            "crs": crs,
            "resolution": resolution,
        },
        result={
            "image_path": str(path),
            "crs": crs,
            "resolution": [resolution, resolution],
        },
        confidence=0.8,
        provenance={
            "image_path": str(path),
            "data_status": "real_validation",
        },
        metadata={
            "sensor": sensor,
            "modality": modality,
        },
    )


def test_compatible_optical_sar_evidence(tmp_path):
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"

    _create_raster(optical)
    _create_raster(sar)

    optical_evidence = _evidence(
        evidence_id="OPT-001",
        sensor="SpaceNet-4",
        modality="optical",
        timestamp="2026-01-20T00:00:00+00:00",
        path=optical,
    )

    sar_evidence = _evidence(
        evidence_id="SAR-001",
        sensor="Sentinel-1",
        modality="sar",
        timestamp="2026-01-20T01:00:00+00:00",
        path=sar,
    )

    result = validate_optical_sar_evidence_compatibility(
        optical_evidence,
        sar_evidence,
    )

    assert result["status"] in {
        "compatible",
        "uncertain",
    }

    assert result["spatial"]["crs_compatible"] is True
    assert result["modalities"] == ["optical", "sar"]
    assert result["sensors"] == [
        "SpaceNet-4",
        "Sentinel-1",
    ]


def test_temporal_separation_is_explicit(tmp_path):
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"

    _create_raster(optical)
    _create_raster(sar)

    optical_evidence = _evidence(
        evidence_id="OPT-002",
        sensor="SpaceNet-4",
        modality="optical",
        timestamp="2026-01-20T00:00:00+00:00",
        path=optical,
    )

    sar_evidence = _evidence(
        evidence_id="SAR-002",
        sensor="Sentinel-1",
        modality="sar",
        timestamp="2026-01-25T00:00:00+00:00",
        path=sar,
    )

    result = validate_optical_sar_evidence_compatibility(
        optical_evidence,
        sar_evidence,
    )

    assert result["temporal"]["timestamp_delta_seconds"] == 432000.0
    assert result["temporal"]["compatible"] is False
    assert result["status"] != "compatible"


def test_different_crs_is_not_spatially_compatible(tmp_path):
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"

    _create_raster(
        optical,
        crs="EPSG:32643",
    )

    _create_raster(
        sar,
        crs="EPSG:32644",
    )

    optical_evidence = _evidence(
        evidence_id="OPT-003",
        sensor="SpaceNet-4",
        modality="optical",
        timestamp="2026-01-20T00:00:00+00:00",
        path=optical,
        crs="EPSG:32643",
    )

    sar_evidence = _evidence(
        evidence_id="SAR-003",
        sensor="Sentinel-1",
        modality="sar",
        timestamp="2026-01-20T00:00:00+00:00",
        path=sar,
        crs="EPSG:32644",
    )

    result = validate_optical_sar_evidence_compatibility(
        optical_evidence,
        sar_evidence,
    )

    assert result["spatial"]["crs_compatible"] is False
    assert result["status"] == "incompatible"


def test_missing_timestamp_becomes_uncertain(tmp_path):
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"

    _create_raster(optical)
    _create_raster(sar)

    optical_evidence = _evidence(
        evidence_id="OPT-004",
        sensor="SpaceNet-4",
        modality="optical",
        timestamp=None,
        path=optical,
    )

    sar_evidence = _evidence(
        evidence_id="SAR-004",
        sensor="Sentinel-1",
        modality="sar",
        timestamp="2026-01-20T00:00:00+00:00",
        path=sar,
    )

    result = validate_optical_sar_evidence_compatibility(
        optical_evidence,
        sar_evidence,
    )

    assert result["temporal"]["compatible"] is None
    assert result["status"] == "uncertain"
