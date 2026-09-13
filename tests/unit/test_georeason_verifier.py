import pytest

from src.schemas.evidence import Evidence
from src.schemas.verification import VerificationResult
from src.verifier.georeason_verifier import GeoReasonVerifier


def make_evidence(
    evidence_id="E1",
    task="building_detection",
    confidence=0.90,
    modality="optical",
    result=None,
):
    return Evidence(
        evidence_id=evidence_id,
        source="test",
        task=task,
        model="test-model",
        sensor="test-sensor",
        modality=modality,
        result=result,
        confidence=confidence,
    )


def test_verifier_accepts_sufficient_high_confidence_evidence():
    verifier = GeoReasonVerifier()

    evidence = [
        make_evidence(
            evidence_id="E1",
            confidence=0.90,
            result={"detected": True},
        )
    ]

    result = verifier.verify(evidence)

    assert isinstance(result, VerificationResult)
    assert result.status == "verified"
    assert result.verified is True
    assert result.confidence == pytest.approx(0.90)
    assert result.evidence_ids == ["E1"]
    assert result.recommended_action == "accept"


def test_verifier_abstains_when_no_evidence_exists():
    verifier = GeoReasonVerifier()

    result = verifier.verify([])

    assert result.status == "abstain"
    assert result.verified is False
    assert result.confidence == 0.0
    assert result.evidence_ids == []
    assert result.recommended_action == "abstain"


def test_verifier_abstains_when_required_modality_is_missing():
    verifier = GeoReasonVerifier()

    evidence = [
        make_evidence(
            evidence_id="E1",
            modality="optical",
            confidence=0.95,
        )
    ]

    result = verifier.verify(
        evidence,
        required_modalities=["sar"],
    )

    assert result.status == "abstain"
    assert result.verified is False
    assert "sar" in result.reasons[0]
    assert result.recommended_action == "abstain"


def test_verifier_abstains_when_expected_task_is_missing():
    verifier = GeoReasonVerifier()

    evidence = [
        make_evidence(
            evidence_id="E1",
            task="water_detection",
            confidence=0.95,
        )
    ]

    result = verifier.verify(
        evidence,
        expected_task="building_detection",
    )

    assert result.status == "abstain"
    assert result.verified is False
    assert any(
        "building_detection" in reason
        for reason in result.reasons
    )
    assert result.recommended_action == "collect_task_specific_evidence"


def test_verifier_reports_low_confidence():
    verifier = GeoReasonVerifier(
        minimum_confidence=0.60,
    )

    evidence = [
        make_evidence(
            evidence_id="E1",
            confidence=0.40,
        )
    ]

    result = verifier.verify(evidence)

    assert result.status == "low_confidence"
    assert result.verified is False
    assert result.confidence == pytest.approx(0.40)
    assert result.recommended_action == "request_additional_evidence"


def test_verifier_detects_conflicting_boolean_evidence():
    verifier = GeoReasonVerifier()

    evidence = [
        make_evidence(
            evidence_id="E1",
            confidence=0.90,
            result={"detected": True},
        ),
        make_evidence(
            evidence_id="E2",
            confidence=0.90,
            result={"detected": False},
        ),
    ]

    result = verifier.verify(evidence)

    assert result.status == "abstain"
    assert result.verified is False
    assert len(result.conflicts) == 1
    assert "contradictory boolean evidence" in result.conflicts[0]
    assert "E1=True" in result.conflicts[0]
    assert "E2=False" in result.conflicts[0]


def test_verifier_accepts_consistent_multiple_evidence_items():
    verifier = GeoReasonVerifier()

    evidence = [
        make_evidence(
            evidence_id="E1",
            confidence=0.80,
            result={"detected": True},
        ),
        make_evidence(
            evidence_id="E2",
            confidence=0.90,
            result={"detected": True},
        ),
    ]

    result = verifier.verify(evidence)

    assert result.status == "verified"
    assert result.verified is True
    assert result.confidence == pytest.approx(0.85)
    assert result.conflicts == []


def test_verifier_supports_multiple_required_modalities():
    verifier = GeoReasonVerifier()

    evidence = [
        make_evidence(
            evidence_id="OPT1",
            modality="optical",
            confidence=0.90,
        ),
        make_evidence(
            evidence_id="SAR1",
            modality="sar",
            confidence=0.80,
        ),
    ]

    result = verifier.verify(
        evidence,
        required_modalities=["optical", "sar"],
    )

    assert result.status == "verified"
    assert result.verified is True
    assert result.confidence == pytest.approx(0.85)


def test_verifier_rejects_invalid_minimum_confidence():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        GeoReasonVerifier(minimum_confidence=1.5)


def test_verifier_rejects_invalid_minimum_evidence():
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        GeoReasonVerifier(minimum_evidence=0)
