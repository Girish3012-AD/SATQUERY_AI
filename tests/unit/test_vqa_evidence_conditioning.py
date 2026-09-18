from pathlib import Path

import pytest

from src.executor.vqa_specialist import VqaSpecialist


class TestVqaEvidenceConditioning:
    """Contract tests for evidence-conditioned VQA."""

    def test_evidence_parameter_accepts_non_empty_string(self):
        evidence = (
            "Remote-sensing building-mask evidence: "
            "928 of 4096 pixels are classified as building pixels "
            "(22.66% coverage). "
            "The derived density category is 'high building density'. "
            "The derived scene category is 'mostly built-up'."
        )

        prepared = VqaSpecialist._resolve_evidence(
            {"evidence": evidence}
        )

        assert prepared == evidence

    def test_missing_evidence_returns_none(self):
        prepared = VqaSpecialist._resolve_evidence({})

        assert prepared is None

    def test_empty_evidence_returns_none(self):
        prepared = VqaSpecialist._resolve_evidence(
            {"evidence": "   "}
        )

        assert prepared is None

    def test_evidence_prompt_contains_exact_evidence(self):
        evidence = (
            "928 of 4096 pixels are classified as building pixels "
            "(22.66% coverage)."
        )

        prompt = VqaSpecialist._build_user_prompt(
            query="What is the building coverage?",
            evidence=evidence,
        )

        assert evidence in prompt

    def test_evidence_prompt_contains_query(self):
        query = "What is the building density?"
        evidence = "The derived density category is 'high building density'."

        prompt = VqaSpecialist._build_user_prompt(
            query=query,
            evidence=evidence,
        )

        assert query in prompt

    def test_evidence_prompt_has_explicit_evidence_boundary(self):
        evidence = "The derived density category is 'high building density'."

        prompt = VqaSpecialist._build_user_prompt(
            query="What is the building density?",
            evidence=evidence,
        )

        assert "REMOTE-SENSING EVIDENCE" in prompt
        assert "END REMOTE-SENSING EVIDENCE" in prompt

    def test_no_evidence_preserves_plain_query(self):
        query = "What can you observe in this image?"

        prompt = VqaSpecialist._build_user_prompt(
            query=query,
            evidence=None,
        )

        assert query in prompt
        assert "REMOTE-SENSING EVIDENCE" not in prompt

    def test_empty_query_is_rejected(self):
        with pytest.raises(ValueError):
            VqaSpecialist._build_user_prompt(
                query="   ",
                evidence="some evidence",
            )
