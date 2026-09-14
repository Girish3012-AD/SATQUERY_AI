from __future__ import annotations

from typing import Any

from shapely.geometry.base import BaseGeometry
from pyproj import CRS

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

    @staticmethod
    def _crs_from_evidence(evidence: Evidence) -> CRS | None:
        """Extract CRS metadata from specialist-produced evidence."""

        candidates = []

        if isinstance(evidence.result, dict):
            candidates.append(evidence.result.get("crs"))

        if isinstance(evidence.metadata, dict):
            candidates.append(evidence.metadata.get("crs"))

        if isinstance(evidence.provenance, dict):
            candidates.append(evidence.provenance.get("crs"))

        for value in candidates:
            if value:
                try:
                    return CRS.from_user_input(value)
                except Exception as exc:
                    raise ValueError(
                        f"Evidence '{evidence.evidence_id}' has invalid CRS: "
                        f"{value!r}"
                    ) from exc

        return None

    def _validate_projected_crs(
        self,
        evidence: Evidence,
        operation: str,
    ) -> CRS:
        """Require a projected CRS for metre-based GIS operations."""

        crs = self._crs_from_evidence(evidence)

        if crs is None:
            raise ValueError(
                f"GIS {operation} requires CRS metadata on evidence "
                f"'{evidence.evidence_id}'."
            )

        if not crs.is_projected:
            raise ValueError(
                f"GIS {operation} requires a projected CRS with linear "
                f"units, but evidence '{evidence.evidence_id}' uses "
                f"geographic CRS '{crs.to_string()}'."
            )

        return crs

    def _validate_compatible_crs(
        self,
        first: Evidence,
        second: Evidence,
        operation: str,
    ) -> tuple[CRS, CRS]:
        """Require both GIS inputs to have matching projected CRS."""

        first_crs = self._validate_projected_crs(first, operation)
        second_crs = self._validate_projected_crs(second, operation)

        if first_crs != second_crs:
            raise ValueError(
                f"GIS {operation} requires matching CRS. "
                f"'{first.evidence_id}' uses {first_crs.to_string()}, "
                f"while '{second.evidence_id}' uses "
                f"{second_crs.to_string()}."
            )

        return first_crs, second_crs

    def _validate_distance_parameter(
        self,
        distance_m: float,
    ) -> float:
        """Validate a metre-based buffer distance."""

        distance_m = float(distance_m)

        if distance_m < 0:
            raise ValueError(
                "Buffer distance cannot be negative."
            )

        if not distance_m == distance_m:
            raise ValueError(
                "Buffer distance must be finite."
            )

        return distance_m

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
            crs = self._validate_projected_crs(
                source,
                operation,
            )

            distance_m = self._validate_distance_parameter(
                distance_m
            )

            geometry = self._geometry_from_evidence(source)

            result_geometry = buffer_geometry(
                geometry,
                distance_m,
            )

            return self._make_evidence(
                operation=operation,
                source_evidence=[source],
                result_geometry=result_geometry,
                measurement={
                    "operation": "buffer",
                    "distance_m": distance_m,
                    "area_m2": calculate_area(result_geometry),
                    "crs": crs.to_string(),
                },
            )

        if operation == "intersection":
            if len(geometry_evidence) < 2:
                raise ValueError(
                    "Intersection requires at least two geometry-bearing "
                    "evidence objects."
                )

            first_evidence = geometry_evidence[-2]
            second_evidence = geometry_evidence[-1]

            crs, _ = self._validate_compatible_crs(
                first_evidence,
                second_evidence,
                operation,
            )

            first = self._geometry_from_evidence(first_evidence)
            second = self._geometry_from_evidence(second_evidence)

            result_geometry = intersect_geometries(first, second)

            return self._make_evidence(
                operation=operation,
                source_evidence=geometry_evidence[-2:],
                result_geometry=result_geometry,
                measurement={
                    "operation": "intersection",
                    "area_m2": calculate_area(result_geometry),
                    "is_empty": bool(result_geometry.is_empty),
                    "crs": crs.to_string(),
                },
            )

        if operation == "distance":
            if len(geometry_evidence) < 2:
                raise ValueError(
                    "Distance requires at least two geometry-bearing "
                    "evidence objects."
                )

            first_evidence = geometry_evidence[-2]
            second_evidence = geometry_evidence[-1]

            crs, _ = self._validate_compatible_crs(
                first_evidence,
                second_evidence,
                operation,
            )

            first = self._geometry_from_evidence(first_evidence)
            second = self._geometry_from_evidence(second_evidence)

            distance = calculate_distance(first, second)

            return self._make_evidence(
                operation=operation,
                source_evidence=geometry_evidence[-2:],
                result_geometry=None,
                measurement={
                    "operation": "distance",
                    "distance_m": float(distance),
                    "crs": crs.to_string(),
                },
                result={"distance_m": float(distance)},
            )

        if operation == "area":
            source = geometry_evidence[-1]
            crs = self._validate_projected_crs(
                source,
                operation,
            )

            geometry = self._geometry_from_evidence(source)

            area = calculate_area(geometry)

            return self._make_evidence(
                operation=operation,
                source_evidence=[source],
                result_geometry=geometry,
                measurement={
                    "operation": "area",
                    "area_m2": float(area),
                    "crs": crs.to_string(),
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
                "crs": (
                    self._crs_from_evidence(source_evidence[0]).to_string()
                    if self._crs_from_evidence(source_evidence[0]) is not None
                    else None
                ),
            },
        )

        self.evidence_registry.add(evidence)

        return evidence
