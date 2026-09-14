from pathlib import Path

import pytest

from src.executor.building_specialist import BuildingDetectionSpecialist
from src.orchestration import SATQueryOrchestrator


IMAGE = Path(
    "data/remote_sensing/spacenet4/"
    "Pan-Sharpen_Atlanta_nadir53_catid_1030010003CD4300_743501_3721539.tif"
)

CHECKPOINT = Path(
    "outputs/checkpoints/building_unet_10epoch_dev.pt"
)

QUERY = "What is the total detected building area in this image?"


@pytest.mark.integration
def test_real_building_area_orchestration():
    if not IMAGE.exists():
        pytest.skip("Real SpaceNet4 image is unavailable.")

    if not CHECKPOINT.exists():
        pytest.skip("BuildingUNet development checkpoint is unavailable.")

    specialist = BuildingDetectionSpecialist(
        checkpoint_path=str(CHECKPOINT)
    )

    orchestrator = SATQueryOrchestrator(
        specialists=[specialist]
    )

    result = orchestrator.run(
        query=QUERY,
        inputs=[str(IMAGE)],
    )

    assert result.task_type == "spatial_analysis"
    assert result.executed_steps == ["T1", "T2", "T3"]
    assert result.successful_steps == ["T1", "T2"]
    assert result.failed_steps == ["T3"]

    assert result.status == "low_confidence"
    assert result.success is False

    assert len(result.evidence_ids) == 2

    evidence = orchestrator.evidence_registry.all()

    assert len(evidence) == 2

    building = next(
        item
        for item in evidence
        if item.task == "building_detection"
    )

    gis_area = next(
        item
        for item in evidence
        if item.task == "gis_area"
    )

    assert building.model == "BuildingUNet_SpaceNet4_dev"
    assert building.modality == "optical"
    assert building.geometry is not None
    assert building.result["crs"] == "EPSG:32616"

    assert building.measurement["building_count"] > 0
    assert building.measurement["predicted_building_pixels"] > 0

    assert gis_area.model == "deterministic-gis"
    assert gis_area.metadata["deterministic"] is True
    assert gis_area.metadata["crs"] == "EPSG:32616"

    assert gis_area.result["area_m2"] > 0
    assert gis_area.measurement["area_m2"] > 0

    assert gis_area.provenance["source_evidence_ids"] == [
        building.evidence_id
    ]

    assert result.verification["status"] == "low_confidence"
    assert result.verification["verified"] is False
    assert result.verification["confidence"] == pytest.approx(
        building.confidence,
        abs=1e-6,
    )

    assert any(
        "below the required threshold" in reason
        for reason in result.verification["reasons"]
    )
