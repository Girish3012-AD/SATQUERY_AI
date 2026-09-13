from __future__ import annotations

from typing import Any

from src.schemas.evidence import Evidence
from src.schemas.verification import VerificationResult


class GeoReasonVerifier:
    """
    Deterministic verifier for SATQuery evidence.

    The verifier does not generate new observations.
    It evaluates whether existing evidence is sufficient,
    consistent, and confident enough to support an answer.
    """

    VERIFIED = "verified"
    LOW_CONFIDENCE = "low_confidence"
    ABSTAIN = "abstain"

    def __init__(
        self,
        minimum_confidence: float = 0.60,
        minimum_evidence: int = 1,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")

        if minimum_evidence < 1:
            raise ValueError("minimum_evidence must be at least 1")

        self.minimum_confidence = minimum_confidence
        self.minimum_evidence = minimum_evidence

    def verify(
        self,
        evidence: list[Evidence],
        *,
        expected_task: str | None = None,
        required_modalities: list[str] | None = None,
    ) -> VerificationResult:
        """
        Verify a collection of evidence items.

        Checks:
        1. Evidence exists.
        2. Evidence has valid confidence.
        3. Expected task is represented when requested.
        4. Required modalities are represented when requested.
        5. Conflicting evidence is detected.
        6. Overall confidence is sufficient.
        """

        required_modalities = required_modalities or []

        if not evidence:
            return VerificationResult(
                status=self.ABSTAIN,
                verified=False,
                confidence=0.0,
                reasons=["No evidence was provided for verification."],
                evidence_ids=[],
                conflicts=[],
                recommended_action="abstain",
            )

        evidence_ids = [item.evidence_id for item in evidence]

        reasons: list[str] = []
        conflicts: list[str] = []

        # ---------------------------------------------------------
        # Confidence validation
        # ---------------------------------------------------------

        confidences = [
            float(item.confidence)
            for item in evidence
        ]

        average_confidence = sum(confidences) / len(confidences)

        invalid_confidence = [
            value
            for value in confidences
            if not 0.0 <= value <= 1.0
        ]

        if invalid_confidence:
            return VerificationResult(
                status=self.ABSTAIN,
                verified=False,
                confidence=0.0,
                reasons=["Evidence contains invalid confidence values."],
                evidence_ids=evidence_ids,
                conflicts=[],
                recommended_action="abstain",
            )

        # ---------------------------------------------------------
        # Minimum evidence requirement
        # ---------------------------------------------------------

        if len(evidence) < self.minimum_evidence:
            return VerificationResult(
                status=self.ABSTAIN,
                verified=False,
                confidence=average_confidence,
                reasons=[
                    "Insufficient evidence for verification."
                ],
                evidence_ids=evidence_ids,
                conflicts=[],
                recommended_action="collect_more_evidence",
            )

        # ---------------------------------------------------------
        # Task validation
        # ---------------------------------------------------------

        if expected_task is not None:
            task_matches = [
                item
                for item in evidence
                if item.task == expected_task
            ]

            if not task_matches:
                reasons.append(
                    f"No evidence supports expected task '{expected_task}'."
                )

        # ---------------------------------------------------------
        # Modality validation
        # ---------------------------------------------------------

        available_modalities = {
            item.modality
            for item in evidence
            if item.modality
        }

        missing_modalities = [
            modality
            for modality in required_modalities
            if modality not in available_modalities
        ]

        if missing_modalities:
            reasons.append(
                "Missing required modalities: "
                + ", ".join(sorted(missing_modalities))
            )

        # ---------------------------------------------------------
        # Conflict detection
        # ---------------------------------------------------------

        conflicts.extend(self._detect_conflicts(evidence))

        if conflicts:
            reasons.append("Conflicting evidence was detected.")

        # ---------------------------------------------------------
        # Confidence decision
        # ---------------------------------------------------------

        if average_confidence < self.minimum_confidence:
            reasons.append(
                "Average evidence confidence is below the "
                f"required threshold of {self.minimum_confidence:.2f}."
            )

        # ---------------------------------------------------------
        # Final decision
        # ---------------------------------------------------------

        if conflicts or missing_modalities:
            return VerificationResult(
                status=self.ABSTAIN,
                verified=False,
                confidence=average_confidence,
                reasons=reasons,
                evidence_ids=evidence_ids,
                conflicts=conflicts,
                recommended_action="abstain",
            )

        if expected_task is not None and not any(
            item.task == expected_task for item in evidence
        ):
            return VerificationResult(
                status=self.ABSTAIN,
                verified=False,
                confidence=average_confidence,
                reasons=reasons,
                evidence_ids=evidence_ids,
                conflicts=conflicts,
                recommended_action="collect_task_specific_evidence",
            )

        if average_confidence < self.minimum_confidence:
            return VerificationResult(
                status=self.LOW_CONFIDENCE,
                verified=False,
                confidence=average_confidence,
                reasons=reasons,
                evidence_ids=evidence_ids,
                conflicts=conflicts,
                recommended_action="request_additional_evidence",
            )

        reasons.append(
            "Evidence satisfies the configured verification criteria."
        )

        return VerificationResult(
            status=self.VERIFIED,
            verified=True,
            confidence=average_confidence,
            reasons=reasons,
            evidence_ids=evidence_ids,
            conflicts=conflicts,
            recommended_action="accept",
        )

    @staticmethod
    def _detect_conflicts(
        evidence: list[Evidence],
    ) -> list[str]:
        """
        Detect simple semantic conflicts between evidence items.

        Evidence with explicit boolean results is compared when
        they refer to the same task.
        """

        conflicts: list[str] = []

        by_task: dict[str, list[Evidence]] = {}

        for item in evidence:
            by_task.setdefault(item.task, []).append(item)

        for task, items in by_task.items():
            boolean_results: list[tuple[str, bool]] = []

            for item in items:
                result = item.result

                if isinstance(result, bool):
                    boolean_results.append(
                        (item.evidence_id, result)
                    )

                elif isinstance(result, dict):
                    value = result.get("detected")

                    if isinstance(value, bool):
                        boolean_results.append(
                            (item.evidence_id, value)
                        )

            values = {value for _, value in boolean_results}

            if len(values) > 1:
                conflicts.append(
                    f"Task '{task}' contains contradictory boolean "
                    "evidence: "
                    + ", ".join(
                        f"{evidence_id}={value}"
                        for evidence_id, value in boolean_results
                    )
                )

        return conflicts
