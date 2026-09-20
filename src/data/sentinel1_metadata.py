from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Sentinel1MetadataResolver:
    """
    Resolve acquisition metadata into the canonical SATQuery
    specialist parameter contract.

    The resolver is intentionally separate from SARSpecialist.
    It converts Sentinel-1 acquisition/STAC metadata into parameters
    that the generic SAR specialist can consume.
    """

    CANONICAL_SENSOR = "Sentinel-1"
    CANONICAL_MODALITY = "sar"
    FREQUENCY_BAND = "C"
    PRODUCT_TYPE = "GRD"

    @classmethod
    def from_json(
        cls,
        metadata_path: str | Path,
    ) -> dict[str, str | bool]:
        path = Path(metadata_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Sentinel-1 metadata file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Sentinel-1 metadata path is not a file: {path}"
            )

        data: dict[str, Any] = json.loads(
            path.read_text(encoding="utf-8")
        )

        return cls.from_dict(data)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, str | bool]:
        if not isinstance(data, dict):
            raise ValueError(
                "Sentinel-1 metadata must be a dictionary."
            )

        mission = data.get("mission")

        if mission != "Sentinel-1":
            raise ValueError(
                f"Expected Sentinel-1 metadata, got mission={mission!r}."
            )

        modality = data.get("modality")

        if modality != "sar":
            raise ValueError(
                f"Expected SAR modality, got modality={modality!r}."
            )

        polarization = data.get("selected_polarization")

        if not polarization:
            raise ValueError(
                "Sentinel-1 metadata is missing selected_polarization."
            )

        data_status = data.get("data_status")

        if not data_status:
            raise ValueError(
                "Sentinel-1 metadata is missing data_status."
            )

        risat_validated = bool(
            data.get("risat_validated", False)
        )

        return {
            "sensor": cls.CANONICAL_SENSOR,
            "modality": cls.CANONICAL_MODALITY,
            "polarization": str(polarization),
            "band": cls.FREQUENCY_BAND,
            "frequency_band": cls.FREQUENCY_BAND,
            "product_type": cls.PRODUCT_TYPE,
            "data_status": str(data_status),
            "units": "source_product_units",
            "timestamp": str(data.get("datetime", "")) or None,
            "risat_validated": risat_validated,
        }


def resolve_sentinel1_parameters(
    metadata_path: str | Path,
) -> dict[str, str | bool]:
    """
    Convenience wrapper for orchestration/input integration.
    """
    return Sentinel1MetadataResolver.from_json(metadata_path)
