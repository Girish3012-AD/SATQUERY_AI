from pathlib import Path

import pytest

from src.controller import TaskController
from src.evidence import EvidenceRegistry
from src.executor import ExecutionEngine
from src.executor.sar_specialist import SARSpecialist
from src.orchestration import SATQueryOrchestrator
from src.planner import EvidencePlanner
from src.registry import ModelRegistry, ModelSpec
from src.router import SensorAwareRouter


@pytest.mark.integration
def test_real_sar_specialist_orchestration() -> None:
    """
    End-to-end SAR integration test.

    NOTE:
    sar_dev/sar.tif is an optical-derived synthetic SAR surrogate.
    This test validates the SAR orchestration/evidence plumbing only.
    It must not be presented as RISAT validation.
    """

    sar_path = Path(
        "data/remote_sensing/sar_dev/sar.tif"
    )

    assert sar_path.exists(), (
        f"Expected SAR development raster: {sar_path}"
    )

    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            name="SAR_Deterministic_Characterizer",
            capability="sar_analysis",
            task_types=["specialized_analysis"],
            modalities=["sar"],
            status="AVAILABLE",
            specialist_name="SARSpecialist",
            checkpoint=None,
            version="deterministic-v1",
            metadata={
                "framework": "rasterio-numpy",
                "execution": "local",
                "offline": True,
                "remote_sensing_adapted": False,
                "role": "sar_characterization",
                "data_status": "synthetic_development_sar",
                "risat_validated": False,
            },
        )
    )

    specialist = SARSpecialist()

    controller = TaskController()
    planner = EvidencePlanner()
    evidence_registry = EvidenceRegistry()
    engine = ExecutionEngine(
        evidence_registry=evidence_registry
    )

    orchestrator = SATQueryOrchestrator(
        controller=controller,
        planner=planner,
        evidence_registry=evidence_registry,
        engine=engine,
        specialists=[specialist],
        registry=registry,
        router=SensorAwareRouter(registry),
    )

    result = orchestrator.run(
        query="Analyze this SAR image.",
        inputs=[str(sar_path)],
    )

    print()
    print("============================================================")
    print("REAL SAR SPECIALIST ORCHESTRATION")
    print("============================================================")
    print("SUCCESS:", result.success)
    print("STATUS:", result.status)
    print("TASK TYPE:", result.task_type)
    print("EXECUTED STEPS:", result.executed_steps)
    print("SUCCESSFUL STEPS:", result.successful_steps)
    print("FAILED STEPS:", result.failed_steps)
    print("SELECTED CAPABILITIES:", result.selected_capabilities)
    print("SELECTED MODELS:", result.selected_models)
    print("EVIDENCE IDS:", result.evidence_ids)
    print("VERIFICATION:", result.verification)

    assert result.task_type == "specialized_analysis"

    assert result.executed_steps == [
        "T1",
        "T2",
    ]

    assert result.successful_steps == [
        "T1",
        "T2",
    ]

    assert result.failed_steps == []

    assert result.success is True
    assert result.status == "verified"

    assert result.selected_capabilities[
        "sar_analysis"
    ] == "SARSpecialist"

    assert result.selected_models[
        "sar_analysis"
    ] == "SAR_Deterministic_Characterizer"

    assert len(result.evidence_ids) == 1

    evidence = evidence_registry.get(
        result.evidence_ids[0]
    )

    assert evidence.task == "sar_analysis"
    assert evidence.model == (
        "SAR_Deterministic_Characterizer"
    )
    assert evidence.modality == "sar"
    assert evidence.sensor == "unknown"

    assert evidence.confidence == pytest.approx(
        0.80
    )

    assert evidence.result is not None
    assert evidence.result[
        "analysis_type"
    ] == "sar_characterization"

    assert evidence.result["crs"] == "EPSG:32616"

    assert evidence.measurement is not None
    assert evidence.measurement[
        "valid_pixel_count"
    ] == 900 * 900

    assert evidence.measurement[
        "valid_fraction"
    ] == pytest.approx(1.0)

    assert evidence.provenance[
        "perception_status"
    ] == "CONNECTED"

    assert evidence.metadata[
        "deterministic"
    ] is True

    assert result.verification["status"] == "verified"
    assert result.verification["verified"] is True

    print()
    print("SAR ORCHESTRATION: PASS")
    print(
        "WARNING: Input is synthetic development SAR, "
        "NOT RISAT."
    )
