from __future__ import annotations

from typing import Any

from shapely.geometry.base import BaseGeometry

from src.evidence import EvidenceRegistry
from src.geospatial import (
    buffer_geometry,
    calculate_area,
    calculate_distance,
    intersect_geometries,
)
from src.schemas import Evidence


class GISEvidenceExecutor:
    """
    Deterministic GIS execution over specialist-produced Evidence.

    The executor never invents geometry. It consumes geometry already
    present in registered evidence and produces a new Evidence object
    containing the deterministic GIS result.
    """

    def __init__(self, evidence_registry: EvidenceRegistry) -> None:
        self.evidence_registry = evidence_registry

    def _geometry_from_evidence(self, evidence: Evidence) -> BaseGeometry:
        geometry_data = evidence.geometry

        if geometry_data is None:
            raise ValueError(
                f"Evidence '{evidence.evidence_id}' contains no geometry."
            )

        if isinstance(geometry_data, BaseGeometry):
            return geometry_data

        if isinstance(geometry_data, dict):
            if "geometry" in geometry_data:
                geometry_data = geometry_data["geometry"]

            if isinstance(geometry_data, BaseGeometry):
                return geometry_data

            if isinstance(geometry_data, dict):
                from shapely.geometry import shape

                return shape(geometry_data)

        raise ValueError(
            f"Evidence '{evidence.evidence_id}' contains unsupported "
            "geometry representation."
        )

    def _select_geometry_evidence(
        self,
        source_evidence_ids: list[str] | None = None,
    ) -> list[Evidence]:
        if source_evidence_ids is not None:
            evidence = [
                self.evidence_registry.get(evidence_id)
                for evidence_id in source_evidence_ids
            ]
        else:
            evidence = [
                evidence
                for evidence in self.evidence_registry.all()
                if evidence.geometry is not None
            ]

        for item in evidence:
            if item.geometry is None:
                raise ValueError(
                    f"Evidence '{item.evidence_id}' contains no geometry."
                )

        return evidence

    def execute(
        self,
        operation: str,
        *,
        parameters: dict[str, Any] | None = None,
        source_evidence_ids: list[str] | None = None,
    ) -> Evidence:
        parameters = dict(parameters or {})

        geometry_evidence = self._select_geometry_evidence(
            source_evidence_ids=source_evidence_ids,
        )

        if not geometry_evidence:
            raise ValueError(
                "GIS operation requires at least one geometry-bearing "
                "evidence object."
            )

        operation = operation.lower().strip()

        if operation == "buffer":
            distance_m = parameters.get("distance_m")

            if distance_m is None:
                raise ValueError(
                    "Buffer operation requires 'distance_m'."
                )

            source = geometry_evidence[-1]
            geometry = self._geometry_from_evidence(source)

            result_geometry = buffer_geometry(
                geometry,
                float(distance_m),
            )

            return self._make_evidence(
                operation=operation,
                source_evidence=[source],
                result_geometry=result_geometry,
                measurement={
                    "operation": "buffer",
                    "distance_m": float(distance_m),
                    "area_m2": calculate_area(result_geometry),
                },
            )

        if operation == "intersection":
            if len(geometry_evidence) < 2:
                raise ValueError(
                    "Intersection requires at least two geometry-bearing "
                    "evidence objects."
                )

            first = self._geometry_from_evidence(geometry_evidence[-2])
            second = self._geometry_from_evidence(geometry_evidence[-1])

            result_geometry = intersect_geometries(first, second)

            return self._make_evidence(
                operation=operation,
                source_evidence=geometry_evidence[-2:],
                result_geometry=result_geometry,
                measurement={
                    "operation": "intersection",
                    "area_m2": calculate_area(result_geometry),
                    "is_empty": bool(result_geometry.is_empty),
                },
            )

        if operation == "distance":
            if len(geometry_evidence) < 2:
                raise ValueError(
                    "Distance requires at least two geometry-bearing "
                    "evidence objects."
                )

            first = self._geometry_from_evidence(geometry_evidence[-2])
            second = self._geometry_from_evidence(geometry_evidence[-1])

            distance = calculate_distance(first, second)

            return self._make_evidence(
                operation=operation,
                source_evidence=geometry_evidence[-2:],
                result_geometry=None,
                measurement={
                    "operation": "distance",
                    "distance_m": float(distance),
                },
                result={"distance_m": float(distance)},
            )

        if operation == "area":
            source = geometry_evidence[-1]
            geometry = self._geometry_from_evidence(source)

            area = calculate_area(geometry)

            return self._make_evidence(
                operation=operation,
                source_evidence=[source],
                result_geometry=geometry,
                measurement={
                    "operation": "area",
                    "area_m2": float(area),
                },
                result={"area_m2": float(area)},
            )

        raise ValueError(
            f"Unsupported GIS operation: {operation}"
        )

    def _make_evidence(
        self,
        *,
        operation: str,
        source_evidence: list[Evidence],
        result_geometry: BaseGeometry | None,
        measurement: dict[str, Any],
        result: Any = None,
    ) -> Evidence:
        source_ids = [
            evidence.evidence_id
            for evidence in source_evidence
        ]

        confidence = min(
            evidence.confidence
            for evidence in source_evidence
        )

        geometry_payload = None

        if result_geometry is not None:
            geometry_payload = {
                "type": "geojson",
                "geometry": result_geometry.__geo_interface__,
            }

        if result is None:
            result = {
                "operation": operation,
                "source_evidence_ids": source_ids,
            }

        evidence = Evidence(
            evidence_id=(
                f"E-GIS-{operation.upper()}-"
                f"{self.evidence_registry.count() + 1:04d}"
            ),
            source="deterministic_gis",
            task=f"gis_{operation}",
            model="deterministic-gis",
            modality="geospatial",
            geometry=geometry_payload,
            measurement=measurement,
            result=result,
            confidence=confidence,
            provenance={
                "operation": operation,
                "source_evidence_ids": source_ids,
            },
            metadata={
                "deterministic": True,
            },
        )

        self.evidence_registry.add(evidence)

        return evidence
