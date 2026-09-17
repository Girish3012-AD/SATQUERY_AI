from __future__ import annotations

import time
from uuid import uuid4

from src.evidence.registry import EvidenceRegistry
from src.geospatial.multimodal_alignment import (
    validate_optical_sar_evidence_compatibility,
)
from src.schemas.evidence import Evidence


class MultimodalEvidenceExecutor:
    """
    Executes deterministic optical + SAR evidence compatibility checks.

    The executor does not perform learned perception.
    It validates whether already-produced optical and SAR Evidence
    can be jointly used for downstream reasoning.
    """

    CAPABILITY = "multimodal_alignment"
    OPERATION = "multimodal_alignment"
    MODEL_NAME = "Deterministic_Multimodal_Alignment"

    def __init__(self, evidence_registry: EvidenceRegistry) -> None:
        self.evidence_registry = evidence_registry

    def execute(
        self,
        source_evidence_ids: list[str] | None = None,
        parameters: dict | None = None,
    ) -> Evidence:
        start_time = time.perf_counter()

        source_ids = list(source_evidence_ids or [])

        if len(source_ids) != 2:
            raise ValueError(
                "Multimodal alignment requires exactly two "
                "dependency Evidence IDs: optical and SAR."
            )

        source_evidence = [
            self.evidence_registry.get(evidence_id)
            for evidence_id in source_ids
        ]

        optical_candidates = [
            evidence
            for evidence in source_evidence
            if evidence.modality == "optical"
        ]

        sar_candidates = [
            evidence
            for evidence in source_evidence
            if evidence.modality == "sar"
        ]

        if len(optical_candidates) != 1:
            raise ValueError(
                "Multimodal alignment requires exactly one "
                "optical Evidence source."
            )

        if len(sar_candidates) != 1:
            raise ValueError(
                "Multimodal alignment requires exactly one "
                "SAR Evidence source."
            )

        optical_evidence = optical_candidates[0]
        sar_evidence = sar_candidates[0]

        compatibility = (
            validate_optical_sar_evidence_compatibility(
                optical_evidence,
                sar_evidence,
            )
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        confidence = min(
            optical_evidence.confidence,
            sar_evidence.confidence,
        )

        evidence_id = (
            "MM_"
            f"{uuid4().hex[:12]}"
        )

        result = {
            "analysis_type": "multimodal_alignment",
            "status": compatibility["status"],
            "compatibility": compatibility,
            "source_evidence_ids": source_ids,
        }

        provenance = {
            "source_evidence_ids": source_ids,
            "executor": self.__class__.__name__,
            "model": self.MODEL_NAME,
            "perception_status": "DETERMINISTIC_VALIDATION",
            "confidence_calibration": "INHERITED_MINIMUM",
        }

        metadata = {
            "deterministic": True,
            "multimodal": True,
            "optical_evidence_id": optical_evidence.evidence_id,
            "sar_evidence_id": sar_evidence.evidence_id,
            "latency_ms": latency_ms,
        }

        return Evidence(
            evidence_id=evidence_id,
            source=self.__class__.__name__,
            task=self.CAPABILITY,
            model=self.MODEL_NAME,
            sensor="multimodal",
            modality="multimodal",
            timestamp=None,
            geometry=None,
            measurement={
                "status": compatibility["status"],
                "source_evidence_ids": source_ids,
            },
            result=result,
            confidence=confidence,
            provenance=provenance,
            metadata=metadata,
        )
