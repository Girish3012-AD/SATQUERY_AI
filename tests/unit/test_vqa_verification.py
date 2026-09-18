import pytest

from src.schemas.evidence import Evidence
from src.schemas.verification import VerificationResult
from src.verifier.georeason_verifier import GeoReasonVerifier


def make_vqa_evidence(confidence=0.90):
    return Evidence(
        evidence_id="VQA-E1",
        source="VqaSpecialist",
        task="vqa",
        model="Qwen2-VL-2B-Instruct",
        sensor=None,
        modality="optical",
        result={
            "answer": (
                "The patch has 22.66% building coverage, "
                "high building density, and a mostly built-up scene."
            )
        },
        confidence=confidence,
        provenance={
            "evidence_conditioned": True,
            "adapter_loaded": True,
        },
        metadata={
            "evidence_conditioned": True,
        },
    )


def test_fully_supported_vqa_claims():
    verifier = GeoReasonVerifier(minimum_confidence=0.60)

    consistency = {
        "numeric_match": True,
        "density_match": True,
        "coverage_match": True,
        "scene_match": True,
    }

    result = verifier.verify(
        [make_vqa_evidence()],
        expected_task="vqa",
        vqa_consistency=consistency,
    )

    assert isinstance(result, VerificationResult)
    assert result.vqa_claim_status == "SUPPORTED"
    assert result.status == "verified"
    assert result.verified is True
    assert result.confidence == pytest.approx(0.90)
    assert result.conflicts == []


def test_partial_vqa_claim_is_distinguished_from_full_support():
    verifier = GeoReasonVerifier(minimum_confidence=0.60)

    consistency = {
        "numeric_match": True,
        "density_match": None,
        "coverage_match": False,
        "scene_match": None,
    }

    result = verifier.verify(
        [make_vqa_evidence()],
        expected_task="vqa",
        vqa_consistency=consistency,
    )

    assert result.vqa_claim_status == "PARTIALLY_SUPPORTED"
    assert result.status == "verified"
    assert result.verified is True
    assert any(
        "partially supported" in reason.lower()
        for reason in result.reasons
    )


def test_numeric_vqa_contradiction_abstains():
    verifier = GeoReasonVerifier(minimum_confidence=0.60)

    consistency = {
        "numeric_match": False,
        "density_match": True,
        "coverage_match": None,
        "scene_match": None,
    }

    result = verifier.verify(
        [make_vqa_evidence()],
        expected_task="vqa",
        vqa_consistency=consistency,
    )

    assert result.vqa_claim_status == "CONTRADICTED"
    assert result.status == "abstain"
    assert result.verified is False
    assert result.conflicts
    assert result.recommended_action == "abstain"


def test_missing_vqa_consistency_does_not_create_claim_status():
    verifier = GeoReasonVerifier(minimum_confidence=0.60)

    result = verifier.verify(
        [make_vqa_evidence()],
        expected_task="vqa",
        vqa_consistency=None,
    )

    assert result.vqa_claim_status is None
    assert result.status == "verified"
    assert result.verified is True


def test_invalid_vqa_consistency_is_not_treated_as_support():
    verifier = GeoReasonVerifier(minimum_confidence=0.60)

    consistency = {
        "numeric_match": "true",
        "density_match": True,
        "coverage_match": None,
        "scene_match": None,
    }

    result = verifier.verify(
        [make_vqa_evidence()],
        expected_task="vqa",
        vqa_consistency=consistency,
    )

    assert result.vqa_claim_status is None
    assert any(
        "invalid vqa consistency" in reason.lower()
        for reason in result.reasons
    )


def test_vqa_claim_status_does_not_lower_existing_confidence_threshold():
    verifier = GeoReasonVerifier(minimum_confidence=0.60)

    consistency = {
        "numeric_match": True,
        "density_match": True,
        "coverage_match": True,
        "scene_match": True,
    }

    result = verifier.verify(
        [make_vqa_evidence(confidence=0.50)],
        expected_task="vqa",
        vqa_consistency=consistency,
    )

    assert result.vqa_claim_status == "SUPPORTED"
    assert result.status == "low_confidence"
    assert result.verified is False
    assert result.confidence == pytest.approx(0.50)
    assert result.recommended_action == "request_additional_evidence"
