from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.data.sentinel1_stac import (
    Sentinel1Candidate,
    Sentinel1STACDiscovery,
)


def test_candidate_serialization() -> None:
    candidate = Sentinel1Candidate(
        item_id="test-item",
        collection="sentinel-1-grd",
        datetime="2026-01-01T00:00:00+00:00",
        geometry=None,
        bbox=[70.0, 20.0, 70.1, 20.1],
        assets={
            "VV": "https://example.com/vv.tif",
            "VH": "https://example.com/vh.tif",
        },
        properties={},
    )

    data = candidate.to_dict()

    assert data["item_id"] == "test-item"
    assert data["collection"] == "sentinel-1-grd"
    assert "VV" in data["assets"]
    assert "VH" in data["assets"]


def test_invalid_bbox() -> None:
    discovery = object.__new__(Sentinel1STACDiscovery)

    with pytest.raises(ValueError):
        discovery.search(
            bbox=[70.0, 20.0, 69.0, 21.0],
            datetime_range="2026-01-01/2026-01-31",
        )


def test_invalid_bbox_length() -> None:
    discovery = object.__new__(Sentinel1STACDiscovery)

    with pytest.raises(ValueError):
        discovery.search(
            bbox=[70.0, 20.0, 70.1],
            datetime_range="2026-01-01/2026-01-31",
        )


def test_extract_vv_vh_assets() -> None:
    discovery = object.__new__(Sentinel1STACDiscovery)

    item = SimpleNamespace(
        assets={
            "vv": SimpleNamespace(
                roles=["data"],
                title="VV polarization",
                href="vv.tif",
            ),
            "vh": SimpleNamespace(
                roles=["data"],
                title="VH polarization",
                href="vh.tif",
            ),
            "thumbnail": SimpleNamespace(
                roles=["thumbnail"],
                title="Preview",
                href="preview.jpg",
            ),
        }
    )

    assets = discovery._extract_sar_assets(item)

    assert assets["VV"] == "vv.tif"
    assert assets["VH"] == "vh.tif"
    assert "thumbnail" not in assets
