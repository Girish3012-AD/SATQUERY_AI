from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .sentinel1_metadata import resolve_sentinel1_parameters


from pydantic import BaseModel, Field

class AssetProfile(BaseModel):
    """Structured classification of an input asset."""
    path: str
    modality: str = "unknown"
    temporal_tag: str | None = None
    bands: list[str] = Field(default_factory=list)
    sensor: str | None = None
    metadata_source: str | None = None


SENTINEL1_METADATA_DIR = Path(
    "data/remote_sensing/sentinel1/metadata"
)


class InputMetadataResolver:
    """
    Resolve known local Earth-observation inputs into canonical
    SATQuery execution parameters and classify assets for routing.
    """

    def __init__(
        self,
        sentinel1_metadata_dir: str | Path = SENTINEL1_METADATA_DIR,
    ) -> None:
        self.sentinel1_metadata_dir = Path(
            sentinel1_metadata_dir
        )

    @staticmethod
    def _metadata_local_path(
        metadata: dict[str, Any],
    ) -> str | None:
        local_path = metadata.get("local_path")

        if local_path is None:
            return None

        return str(local_path)

    def _find_sentinel1_metadata(
        self,
        input_path: str | Path,
    ) -> Path | None:
        input_path = Path(input_path).resolve()

        if not self.sentinel1_metadata_dir.exists():
            return None

        for metadata_path in sorted(
            self.sentinel1_metadata_dir.glob("*.json")
        ):
            try:
                metadata = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError):
                continue

            mission = metadata.get("mission")

            if mission != "Sentinel-1":
                continue

            local_path = self._metadata_local_path(
                metadata
            )

            if local_path is None:
                continue

            if Path(local_path).resolve() == input_path:
                return metadata_path

        return None

    def classify_assets(self, inputs: list[str]) -> list[AssetProfile]:
        """
        Classify a list of input paths into typed AssetProfiles.
        First tries JSON metadata, then falls back to deterministic filename parsing.
        """
        profiles = []
        for input_path in inputs:
            profile = AssetProfile(path=input_path)
            
            # Check explicit Sentinel-1 JSON metadata first
            s1_metadata_path = self._find_sentinel1_metadata(input_path)
            if s1_metadata_path:
                profile.modality = "sar"
                profile.sensor = "Sentinel-1"
                profile.metadata_source = str(s1_metadata_path)
            else:
                # Deterministic filename fallback for untracked assets
                path_upper = Path(input_path).name.upper()
                
                if "S1" in path_upper or "SAR" in path_upper or "GRD" in path_upper:
                    profile.modality = "sar"
                    profile.sensor = "Sentinel-1"
                elif "S2" in path_upper or "OPTICAL" in path_upper or "B03" in path_upper or "B08" in path_upper:
                    profile.modality = "optical"
                    profile.sensor = "Sentinel-2"
                
                if "B03" in path_upper:
                    profile.bands.append("B03")
                if "B08" in path_upper:
                    profile.bands.append("B08")
                
                if "T1" in path_upper:
                    profile.temporal_tag = "T1"
                elif "T2" in path_upper:
                    profile.temporal_tag = "T2"

                profile.metadata_source = "filename_fallback"

            profiles.append(profile)
            
        return profiles

    def resolve(
        self,
        inputs: list[str],
    ) -> dict[str, str | int | float | bool]:
        """
        Resolve execution parameters from the supplied inputs.

        Parameters are only added when the input has validated
        acquisition provenance. No metadata means no inferred
        specialist parameters.
        """

        resolved: dict[str, str | int | float | bool] = {}

        for input_path in inputs:
            metadata_path = self._find_sentinel1_metadata(
                input_path
            )

            if metadata_path is None:
                continue

            sentinel1_parameters = (
                resolve_sentinel1_parameters(
                    metadata_path
                )
            )

            resolved.update(
                sentinel1_parameters
            )

            # Current validation uses one SAR scene.
            # Stop once a matching Sentinel-1 input is found.
            break

        return resolved
