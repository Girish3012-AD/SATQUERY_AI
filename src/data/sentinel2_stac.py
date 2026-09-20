"""
sentinel2_stac.py — Sentinel-2 scene discovery through STAC.

Mirrors the architecture of sentinel1_stac.py.
Uses pystac-client to query the Microsoft Planetary Computer catalog.

No hard-coded scenes. Raises clear errors when no data is found.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from pystac_client import Client


PLANETARY_COMPUTER_STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

SENTINEL2_COLLECTION = "sentinel-2-l2a"

# Bands used by SATQuery spectral analysis
REQUIRED_BANDS = ("B03", "B08")
OPTIONAL_BANDS = ("B02", "B04")


@dataclass
class Sentinel2Candidate:
    """Metadata for a single Sentinel-2 scene candidate."""

    item_id: str
    collection: str
    datetime: str | None
    geometry: dict[str, Any] | None
    bbox: list[float] | None
    assets: dict[str, str]        # band name → asset href
    cloud_cover: float | None     # percent, None if unavailable
    platform: str | None
    properties: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Sentinel2STACDiscovery:
    """
    Reproducible Sentinel-2 L2A discovery through STAC.

    Retrieves real scenes for a requested AOI and date range.
    Does not fabricate scene identifiers, dates, or coordinates.
    """

    def __init__(
        self,
        catalog_url: str = PLANETARY_COMPUTER_STAC_URL,
    ) -> None:
        self.catalog_url = catalog_url
        self.catalog = Client.open(catalog_url)

    @staticmethod
    def _extract_band_assets(
        item: Any,
    ) -> dict[str, str]:
        """
        Extract band asset hrefs from a STAC item.

        Sentinel-2 L2A items on Planetary Computer expose band assets
        with keys matching the band name (e.g. "B03", "B08").
        Only bands present in the item are included.
        """
        assets: dict[str, str] = {}

        target_bands = set(REQUIRED_BANDS) | set(OPTIONAL_BANDS)

        for key, asset in item.assets.items():
            key_upper = str(key).upper()
            if key_upper in target_bands:
                assets[key_upper] = asset.href

        return assets

    @staticmethod
    def _cloud_cover(item: Any) -> float | None:
        props = item.properties or {}
        for field in ("eo:cloud_cover", "cloud_cover", "cloudcover"):
            value = props.get(field)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        return None

    def search(
        self,
        bbox: list[float],
        datetime_range: str,
        max_items: int = 20,
        max_cloud_pct: float | None = 80.0,
    ) -> list[Sentinel2Candidate]:
        """
        Search for Sentinel-2 scenes.

        Parameters
        ----------
        bbox : [min_lon, min_lat, max_lon, max_lat]
        datetime_range : ISO 8601 interval, e.g. "2023-01-01/2023-03-31"
        max_items : search limit
        max_cloud_pct : filter out scenes with more cloud cover than this

        Returns
        -------
        List of candidates sorted by cloud cover ascending.
        """
        if len(bbox) != 4:
            raise ValueError(
                "bbox must contain [min_lon, min_lat, max_lon, max_lat]."
            )

        min_lon, min_lat, max_lon, max_lat = bbox

        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("Invalid bbox: min must be less than max.")

        search = self.catalog.search(
            collections=[SENTINEL2_COLLECTION],
            bbox=bbox,
            datetime=datetime_range,
            max_items=max_items,
        )

        candidates: list[Sentinel2Candidate] = []

        for item in search.items():
            cloud = self._cloud_cover(item)

            if (
                max_cloud_pct is not None
                and cloud is not None
                and cloud > max_cloud_pct
            ):
                continue

            assets = self._extract_band_assets(item)

            candidates.append(
                Sentinel2Candidate(
                    item_id=item.id,
                    collection=SENTINEL2_COLLECTION,
                    datetime=(
                        item.datetime.isoformat()
                        if item.datetime is not None
                        else None
                    ),
                    geometry=item.geometry,
                    bbox=list(item.bbox) if item.bbox else None,
                    assets=assets,
                    cloud_cover=cloud,
                    platform=item.properties.get("platform"),
                    properties=dict(item.properties),
                )
            )

        # Sort: prefer low cloud cover, then by date (most recent first)
        candidates.sort(
            key=lambda c: (
                c.cloud_cover if c.cloud_cover is not None else 100.0,
                -(
                    _iso_sort_key(c.datetime)
                    if c.datetime
                    else 0
                ),
            )
        )

        return candidates

    def search_best(
        self,
        bbox: list[float],
        datetime_range: str,
        *,
        require_bands: tuple[str, ...] = REQUIRED_BANDS,
    ) -> Sentinel2Candidate:
        """
        Return the best available scene for the AOI.

        "Best" = lowest cloud cover with required bands present.
        Raises RuntimeError when no usable scene is found.
        """
        candidates = self.search(
            bbox=bbox,
            datetime_range=datetime_range,
            max_items=50,
        )

        if not candidates:
            raise RuntimeError(
                f"No Sentinel-2 scenes found for bbox={bbox} "
                f"in date range '{datetime_range}'."
            )

        usable = [
            c for c in candidates
            if all(band in c.assets for band in require_bands)
        ]

        if not usable:
            raise RuntimeError(
                f"Sentinel-2 scenes were found for bbox={bbox}, "
                f"but none contained the required bands "
                f"{list(require_bands)}. "
                f"Found {len(candidates)} scene(s) total."
            )

        return usable[0]


def _iso_sort_key(dt_str: str) -> int:
    """Convert ISO datetime string to sortable integer (unix-like)."""
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(
            dt_str.replace("Z", "+00:00")
        )
        return int(dt.timestamp())
    except Exception:
        return 0
