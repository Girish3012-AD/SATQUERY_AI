from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .sentinel1_metadata import resolve_sentinel1_parameters


SENTINEL1_METADATA_DIR = Path(
    "data/remote_sensing/sentinel1/metadata"
)


class InputMetadataResolver:
    """
    Resolve known local Earth-observation inputs into canonical
    SATQuery execution parameters.

    The resolver currently supports Sentinel-1 provenance metadata.
    Unsupported inputs simply return an empty parameter dictionary.
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
