from pathlib import Path

import pytest

from src.orchestration.orchestrator import SATQueryOrchestrator


@pytest.mark.integration
def test_real_sentinel1_vv_end_to_end_orchestration():

    raster = Path(
        "data/remote_sensing/sentinel1/processed/"
        "S1A_IW_GRDH_1SDV_07E290_VV_PUNE_AOI_EPSG32643_10m.tif"
    )

    if not raster.exists():
        pytest.skip(
            f"Real Sentinel-1 validation raster not available: {raster}"
        )

    query = "Analyze this SAR image."

    orchestrator = SATQueryOrchestrator()

    result = orchestrator.run(
        query=query,
        inputs=[str(raster)],
    )

    # ---------------------------------------------------------
    # ORCHESTRATION
    # ---------------------------------------------------------

    assert result.success is True
    assert result.status == "verified"
    assert result.task_type == "specialized_analysis"

    # ---------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------

    assert "T1" in result.executed_steps
    assert "T1" in result.successful_steps
    assert result.failed_steps == []

    # ---------------------------------------------------------
    # ROUTING
    # ---------------------------------------------------------

    assert (
        result.selected_capabilities["sar_analysis"]
        == "SARSpecialist"
    )

    assert (
        result.selected_models["sar_analysis"]
        == "SAR_Deterministic_Characterizer"
    )

    # ---------------------------------------------------------
    # EVIDENCE
    # ---------------------------------------------------------

    assert len(result.evidence_ids) == 1

    evidence_items = orchestrator.evidence_registry.all()

    sar_evidence = [
        evidence
        for evidence in evidence_items
        if evidence.task == "sar_analysis"
    ]

    assert len(sar_evidence) == 1

    evidence = sar_evidence[0]

    assert evidence.evidence_id in result.evidence_ids

    assert evidence.source == "SARSpecialist"
    assert evidence.task == "sar_analysis"
    assert evidence.model == "SAR_Deterministic_Characterizer"

    # ---------------------------------------------------------
    # SENSOR IDENTITY
    # ---------------------------------------------------------

    assert evidence.sensor == "Sentinel-1"
    assert evidence.modality == "sar"

    # ---------------------------------------------------------
    # SENTINEL-1 PRODUCT CONTRACT
    # ---------------------------------------------------------

    assert evidence.result is not None

    assert evidence.result["crs"] == "EPSG:32643"
    assert evidence.result["polarization"] == "VV"
    assert evidence.result["band"] == "C"
    assert evidence.result["frequency_band"] == "C"
    assert evidence.result["product_type"] == "GRD"

    assert evidence.result["data_status"] == "real_validation"
    assert evidence.result["risat_validated"] is False

    assert evidence.result["has_valid_data"] is True

    # ---------------------------------------------------------
    # MEASUREMENTS
    # ---------------------------------------------------------

    assert evidence.measurement is not None

    assert evidence.measurement["valid_pixel_count"] > 0
    assert evidence.measurement["valid_fraction"] == 1.0

    assert evidence.measurement["resolution_x"] == 10.0
    assert evidence.measurement["resolution_y"] == 10.0

    # ---------------------------------------------------------
    # PROVENANCE
    # ---------------------------------------------------------

    assert evidence.provenance["data_status"] == "real_validation"
    assert evidence.provenance["risat_validated"] is False
    assert evidence.provenance["learned_detection"] is False
    assert evidence.provenance["perception_status"] == "CONNECTED"

    # ---------------------------------------------------------
    # DETERMINISTIC SPECIALIST CONTRACT
    # ---------------------------------------------------------

    assert evidence.metadata["deterministic"] is True
    assert evidence.confidence == 0.80

    # ---------------------------------------------------------
    # VERIFICATION
    # ---------------------------------------------------------

    verification = result.verification

    assert isinstance(verification, dict)

    assert verification["status"] == "verified"
    assert verification["verified"] is True
    assert verification["confidence"] == 0.80

    assert verification["evidence_ids"] == result.evidence_ids
    assert verification["conflicts"] == []

    assert verification["recommended_action"] == "accept"

    assert (
        "Evidence satisfies the configured verification criteria."
        in verification["reasons"]
    )
