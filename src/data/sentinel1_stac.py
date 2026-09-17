from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

from pystac_client import Client


PLANETARY_COMPUTER_STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

SENTINEL1_COLLECTION = "sentinel-1-grd"


@dataclass
class Sentinel1Candidate:
    item_id: str
    collection: str
    datetime: str | None
    geometry: dict[str, Any] | None
    bbox: list[float] | None
    assets: dict[str, str]
    properties: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Sentinel1STACDiscovery:
    """
    Reproducible Sentinel-1 GRD discovery through STAC.

    This component performs discovery only.
    It does not claim RISAT compatibility or perform SAR perception.
    """

    def __init__(
        self,
        catalog_url: str = PLANETARY_COMPUTER_STAC_URL,
    ) -> None:
        self.catalog_url = catalog_url
        self.catalog = Client.open(catalog_url)

    @staticmethod
    def _extract_sar_assets(item: Any) -> dict[str, str]:
        """
        Extract actual Sentinel-1 raster polarization assets.

        Do not identify assets by searching arbitrary href text. A
        calibration XML path can contain "vv" or "vh" even though it
        is not the SAR raster we want.
        """
        assets: dict[str, str] = {}

        for key, asset in item.assets.items():
            key_lower = str(key).lower()
            title = (asset.title or "").lower()
            media_type = (getattr(asset, "media_type", None) or "").lower()
            roles = {
                str(role).lower()
                for role in (asset.roles or [])
            }

            # Prefer STAC asset keys that explicitly represent
            # Sentinel-1 polarization.
            polarization: str | None = None

            if key_lower in {"vv", "vv-gamma0", "vv-gamma0-rtc"}:
                polarization = "VV"
            elif key_lower in {"vh", "vh-gamma0", "vh-gamma0-rtc"}:
                polarization = "VH"

            # Some catalogs expose polarization through title/metadata.
            # Only accept it when the asset is actually a raster/data
            # asset, never merely because an XML calibration path contains
            # the letters "vv" or "vh".
            if polarization is None:
                text = " ".join(
                    [
                        key_lower,
                        title,
                        media_type,
                        *roles,
                    ]
                )

                is_raster = (
                    "tif" in media_type
                    or "tiff" in media_type
                    or "cog" in media_type
                    or "raster" in roles
                    or "data" in roles
                )

                if is_raster:
                    if "vv" in text:
                        polarization = "VV"
                    elif "vh" in text:
                        polarization = "VH"

            if polarization is not None:
                assets[polarization] = asset.href

        return assets

    def search(
        self,
        bbox: list[float],
        datetime_range: str,
        max_items: int = 10,
    ) -> list[Sentinel1Candidate]:
        if len(bbox) != 4:
            raise ValueError(
                "bbox must contain [min_lon, min_lat, max_lon, max_lat]."
            )

        min_lon, min_lat, max_lon, max_lat = bbox

        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("Invalid bbox ordering.")

        search = self.catalog.search(
            collections=[SENTINEL1_COLLECTION],
            bbox=bbox,
            datetime=datetime_range,
            max_items=max_items,
        )

        candidates: list[Sentinel1Candidate] = []

        for item in search.items():
            assets = self._extract_sar_assets(item)

            candidates.append(
                Sentinel1Candidate(
                    item_id=item.id,
                    collection=SENTINEL1_COLLECTION,
                    datetime=(
                        item.datetime.isoformat()
                        if item.datetime is not None
                        else None
                    ),
                    geometry=item.geometry,
                    bbox=list(item.bbox) if item.bbox else None,
                    assets=assets,
                    properties=dict(item.properties),
                )
            )

        candidates.sort(
            key=lambda candidate: (
                "VV" not in candidate.assets,
                "VH" not in candidate.assets,
                candidate.datetime or "",
            )
        )

        return candidates

    def search_best(
        self,
        bbox: list[float],
        datetime_range: str,
    ) -> Sentinel1Candidate:
        candidates = self.search(
            bbox=bbox,
            datetime_range=datetime_range,
            max_items=20,
        )

        if not candidates:
            raise RuntimeError(
                "No Sentinel-1 GRD scenes found for the requested "
                "AOI and time range."
            )

        vv_candidates = [
            candidate
            for candidate in candidates
            if "VV" in candidate.assets
        ]

        if not vv_candidates:
            raise RuntimeError(
                "Sentinel-1 scenes were found, but no VV asset "
                "was available."
            )

        return vv_candidates[0]


def candidate_to_json_dict(
    candidate: Sentinel1Candidate,
) -> dict[str, Any]:
    return candidate.to_dict()
