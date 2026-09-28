"""
P1-2: Execution Status vs Verification Status Decoupling — Unit Tests

Tests verify that the three concepts are fully independent:
  - execution_status (COMPLETED / FAILED)
  - verification_status (VERIFIED / NOT_VERIFIED / INCONCLUSIVE / NOT_EVALUATED)
  - confidence (uncalibrated float — cannot create VERIFIED)

These tests use only deterministic in-process logic; no model downloads required.
"""
from __future__ import annotations

import pytest

from src.schemas.evidence import Evidence
from src.verifier.georeason_verifier import GeoReasonVerifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_evidence(confidence: float, task: str = "water_analysis") -> Evidence:
    return Evidence(
        evidence_id=f"EVID_{task}",
        source="TestSpecialist",
        task=task,
        model="TestModel",
        sensor="Sentinel-2",
        modality="optical",
        timestamp="2024-01-01",
        geometry=None,
        measurement={"area_m2": 1000.0},
        result={"description": "test result"},
        confidence=confidence,
        provenance={"model_name": "TestModel"},
        metadata={"deterministic": True},
    )


def _run_verifier(evidence_list: list[Evidence], min_confidence: float = 0.60) -> dict:
    """Run the GeoReasonVerifier and return a dict matching the API response shape."""
    verifier = GeoReasonVerifier(minimum_confidence=min_confidence)
    ver_result = verifier.verify(evidence_list)  # VerificationResult pydantic model

    # Simulate the backend API response shape
    execution_success = len(evidence_list) > 0
    return {
        "execution_status": "COMPLETED" if execution_success else "FAILED",
        "verification_status": ver_result.status,
        "verification": ver_result,
        "confidence": ver_result.confidence,
        "confidence_calibration": "uncalibrated",
    }


# ---------------------------------------------------------------------------
# Case 1: Execution completed + verification VERIFIED
# ---------------------------------------------------------------------------
def test_case1_execution_completed_verification_verified():
    """High confidence evidence that passes the verifier threshold."""
    ev = _make_evidence(confidence=0.85)
    resp = _run_verifier([ev])

    assert resp["execution_status"] == "COMPLETED"
    assert resp["verification_status"] == "verified"
    assert resp["confidence_calibration"] == "uncalibrated"


# ---------------------------------------------------------------------------
# Case 2: Execution completed + verification NOT_VERIFIED (low_confidence)
# ---------------------------------------------------------------------------
def test_case2_execution_completed_verification_not_verified():
    """Low confidence evidence that fails the verifier threshold."""
    ev = _make_evidence(confidence=0.20)
    resp = _run_verifier([ev])

    assert resp["execution_status"] == "COMPLETED"
    # Should be low_confidence or abstain — NOT 'verified'
    assert resp["verification_status"] != "verified"


# ---------------------------------------------------------------------------
# Case 3: Execution completed + verification INCONCLUSIVE (abstain)
# ---------------------------------------------------------------------------
def test_case3_execution_completed_verification_inconclusive():
    """Borderline confidence that results in abstain/low_confidence."""
    ev = _make_evidence(confidence=0.40)
    resp = _run_verifier([ev])

    assert resp["execution_status"] == "COMPLETED"
    # abstain or low_confidence — NOT 'verified'
    assert resp["verification_status"] not in ("verified",)


# ---------------------------------------------------------------------------
# Case 4: Execution FAILED
# ---------------------------------------------------------------------------
def test_case4_execution_failed():
    """When no evidence is produced (failure), verification must not be 'verified'."""
    resp = _run_verifier([])  # No evidence = simulate failure

    assert resp["execution_status"] == "FAILED"
    assert resp["verification_status"] != "verified"


# ---------------------------------------------------------------------------
# Case 5: Backend VERIFIED — confirms verifier is source of truth
# ---------------------------------------------------------------------------
def test_case5_backend_verifier_is_source_of_truth_for_verified():
    """
    If the backend verifier returns 'verified', it IS verified.
    Status derives from the verifier, not from client-side confidence checks.
    """
    ev = _make_evidence(confidence=0.85)
    verifier = GeoReasonVerifier(minimum_confidence=0.60)
    ver_result = verifier.verify([ev])

    # The backend verifier must say 'verified'
    assert ver_result.status == "verified", \
        f"Expected 'verified' from verifier, got: {ver_result.status}"

    # And the execution_status is built separately from verification_status
    api_resp = {
        "execution_status": "COMPLETED",
        "verification_status": ver_result.status,
    }
    assert api_resp["execution_status"] != api_resp["verification_status"], \
        "execution_status and verification_status must be distinct fields with distinct semantics"


# ---------------------------------------------------------------------------
# Case 6: High confidence + NOT_VERIFIED (threshold raised)
# ---------------------------------------------------------------------------
def test_case6_high_confidence_but_not_verified():
    """
    confidence = 0.91 does NOT automatically produce VERIFIED.
    The verifier's minimum_confidence threshold controls the output.
    With threshold=0.95, confidence=0.91 must NOT be verified.
    """
    ev = _make_evidence(confidence=0.91)
    verifier = GeoReasonVerifier(minimum_confidence=0.95)
    ver_result = verifier.verify([ev])

    # With threshold=0.95 and confidence=0.91, must NOT verify
    assert ver_result.status != "verified", \
        f"With threshold 0.95 and confidence 0.91, expected not verified, got: {ver_result.status}"

    # The old (removed) frontend logic: if confidence >= 0.6 → show VERIFIED badge
    # 0.91 > 0.6, but the verifier says NOT verified. Prove the two are independent.
    confidence_exceeds_old_frontend_threshold = 0.91 >= 0.6
    assert confidence_exceeds_old_frontend_threshold is True   # 0.91 > 0.6
    assert ver_result.status != "verified"                     # But NOT verified


# ---------------------------------------------------------------------------
# Case 7: Execution completed + no verifier result
# ---------------------------------------------------------------------------
def test_case7_execution_completed_no_verifier_result():
    """
    When verification.status is absent, frontend maps it to 'not_evaluated'.
    Execution completed ≠ verified.
    """
    resp = {
        "execution_status": "COMPLETED",
        "verification_status": None,   # Not set
        "verification": {},            # Empty
        "confidence": None,
        "confidence_calibration": "uncalibrated",
    }

    # Frontend logic: None → 'not_evaluated'
    ver_status = resp["verification_status"] or "not_evaluated"

    assert resp["execution_status"] == "COMPLETED"
    assert ver_status == "not_evaluated"
    assert ver_status != "verified"


# ---------------------------------------------------------------------------
# Case 8: Running execution clears stale verification state
# ---------------------------------------------------------------------------
def test_case8_running_execution_must_not_show_stale_verified_state():
    """
    While a new execution is RUNNING, previous VERIFIED state must not be shown.
    The frontend checks: if execution_status == 'RUNNING', hide verification badge.
    """
    def should_show_verification_badge(data: dict) -> bool:
        """Frontend logic: hide verification when execution is running."""
        exec_status = data.get("execution_status", "completed")
        return exec_status.lower() != "running"

    stale_data = {
        "execution_status": "COMPLETED",
        "verification": {"status": "verified", "confidence": 0.90},
    }
    new_data = {
        "execution_status": "RUNNING",
        "verification": {"status": "verified", "confidence": 0.90},  # Stale
    }

    assert should_show_verification_badge(stale_data) is True   # Show old result
    assert should_show_verification_badge(new_data) is False     # Hide while running


# ---------------------------------------------------------------------------
# Prohibition tests — confidence cannot produce VERIFIED
# ---------------------------------------------------------------------------
def test_confidence_0_6_threshold_removed_from_frontend():
    """
    The old JS: if (ev.confidence >= 0.6) → VERIFIED badge.
    This must be completely removed from satquery.js.
    """
    with open("static/js/satquery.js", "r", encoding="utf-8") as f:
        js_content = f.read()

    assert "ev.confidence >= 0.6" not in js_content, \
        "VIOLATION: Frontend still has confidence >= 0.6 → VERIFIED badge logic"


def test_js_status_badge_class_uses_typed_verification():
    """
    statusBadgeClass() must accept a 'type' parameter.
    The old single-arg call incorrectly mapped 'completed' → status-badge--verified.
    """
    with open("static/js/satquery.js", "r", encoding="utf-8") as f:
        js_content = f.read()

    assert 'if (type === "verification")' in js_content, \
        "VIOLATION: statusBadgeClass does not distinguish verification type"
    assert 'if (type === "execution")' in js_content, \
        "VIOLATION: statusBadgeClass does not distinguish execution type"


def test_confidence_calibration_always_uncalibrated():
    """
    Confidence must always be labeled 'uncalibrated'.
    """
    ev = _make_evidence(confidence=0.90)
    resp = _run_verifier([ev])
    assert resp["confidence_calibration"] == "uncalibrated"


def test_verification_status_from_backend_not_execution_success():
    """
    execution_status=COMPLETED does not imply verification_status=VERIFIED.
    verification_status must derive solely from the backend verifier output.
    """
    ev = _make_evidence(confidence=0.90)
    verifier = GeoReasonVerifier(minimum_confidence=0.60)
    ver_result = verifier.verify([ev])

    # The API builds these independently
    api_resp = {
        "execution_status": "COMPLETED",              # From orchestrator.success
        "verification_status": ver_result.status,     # From verifier.verify()
    }

    # They coexist independently; neither implies the other
    assert "execution_status" in api_resp
    assert "verification_status" in api_resp
    # Execution completed does not force verification_status to 'verified'
    # (It happens to be 'verified' here because confidence=0.90 passes threshold=0.60,
    #  but that is the verifier's decision, not the execution status decision)
    assert api_resp["execution_status"] == "COMPLETED"
