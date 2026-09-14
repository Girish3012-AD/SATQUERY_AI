import pytest

from src.evidence import EvidenceRegistry
from src.executor import ExecutionEngine, Specialist
from src.planner import EvidencePlanner
from src.schemas import Evidence, TaskSpec


class FakeBuildingSpecialist(Specialist):

    @property
    def capability(self) -> str:
        return "building_detection"

    def infer(self, inputs, parameters=None):
        return Evidence(
            evidence_id="E-BUILDING-001",
            source="test-fixture",
            task="building_detection",
            model="FakeBuildingSpecialist",
            modality="optical",
            result={
                "detections": [
                    {
                        "label": "building",
                        "confidence": 0.95,
                    }
                ]
            },
            confidence=0.95,
            provenance={
                "test_only": True,
            },
        )


def make_task():
    return TaskSpec(
        task_id="TASK-001",
        query="Find buildings.",
        task_type="specialized_analysis",
        required_capabilities=["building_detection"],
        required_modalities=["optical"],
        input_count=1,
    )


def test_specialist_execution_creates_evidence():
    registry = EvidenceRegistry()
    engine = ExecutionEngine(registry)

    engine.register_specialist(
        FakeBuildingSpecialist()
    )

    task = make_task()
    plan = EvidencePlanner().create_plan(task)

    result = engine.execute_step(
        plan.get_step("T1"),
        inputs=["test.tif"],
    )

    assert result.success is True
    assert result.evidence_ids == ["E-BUILDING-001"]

    evidence = registry.get("E-BUILDING-001")

    assert evidence.task == "building_detection"
    assert evidence.confidence == 0.95


def test_missing_specialist_fails_safely():
    registry = EvidenceRegistry()
    engine = ExecutionEngine(registry)

    task = make_task()
    plan = EvidencePlanner().create_plan(task)

    result = engine.execute_step(
        plan.get_step("T1"),
        inputs=["test.tif"],
    )

    assert result.success is False
    assert result.evidence_ids == []
    assert "No specialist implementation" in result.message


def test_duplicate_specialist_rejected():
    registry = EvidenceRegistry()
    engine = ExecutionEngine(registry)

    engine.register_specialist(
        FakeBuildingSpecialist()
    )

    with pytest.raises(ValueError):
        engine.register_specialist(
            FakeBuildingSpecialist()
        )


def test_full_plan_stops_when_step_fails():
    registry = EvidenceRegistry()
    engine = ExecutionEngine(registry)

    task = make_task()
    plan = EvidencePlanner().create_plan(task)

    results = engine.execute(
        plan,
        inputs=["test.tif"],
    )

    assert len(results) == 1
    assert results[0].success is False
    assert registry.count() == 0


def test_executor_runs_verification_step():
    from src.executor.executor import ExecutionEngine
    from src.planner.evidence_planner import PlanStep
    from src.schemas.evidence import Evidence
    from src.evidence.registry import EvidenceRegistry

    registry = EvidenceRegistry()

    evidence = Evidence(
        evidence_id="E_VERIFY_1",
        source="test",
        task="building_detection",
        model="test-model",
        sensor="test-sensor",
        modality="optical",
        result={"detected": True},
        confidence=0.90,
    )

    registry.add(evidence)

    engine = ExecutionEngine(registry)

    step = PlanStep(
        step_id="T2",
        task="verification",
        operation="verification",
    )

    result = engine.execute_step(step)

    assert result.success is True
    assert result.task == "verification"
    assert result.output["status"] == "verified"
    assert result.output["verified"] is True
    assert result.output["confidence"] == 0.90
    assert result.evidence_ids == ["E_VERIFY_1"]


def test_executor_verification_fails_on_low_confidence():
    from src.executor.executor import ExecutionEngine
    from src.planner.evidence_planner import PlanStep
    from src.schemas.evidence import Evidence
    from src.evidence.registry import EvidenceRegistry

    registry = EvidenceRegistry()

    evidence = Evidence(
        evidence_id="E_LOW_1",
        source="test",
        task="building_detection",
        model="test-model",
        sensor="test-sensor",
        modality="optical",
        result={"detected": True},
        confidence=0.30,
    )

    registry.add(evidence)

    engine = ExecutionEngine(registry)

    step = PlanStep(
        step_id="T2",
        task="verification",
        operation="verification",
    )

    result = engine.execute_step(step)

    assert result.success is False
    assert result.task == "verification"
    assert result.output["status"] == "low_confidence"
    assert result.output["verified"] is False
    assert result.output["recommended_action"] == (
        "request_additional_evidence"
    )


def test_executor_verification_detects_conflicting_evidence():
    from src.executor.executor import ExecutionEngine
    from src.planner.evidence_planner import PlanStep
    from src.schemas.evidence import Evidence
    from src.evidence.registry import EvidenceRegistry

    registry = EvidenceRegistry()

    registry.add(
        Evidence(
            evidence_id="E_TRUE",
            source="test",
            task="building_detection",
            model="model-a",
            sensor="test-sensor",
            modality="optical",
            result={"detected": True},
            confidence=0.90,
        )
    )

    registry.add(
        Evidence(
            evidence_id="E_FALSE",
            source="test",
            task="building_detection",
            model="model-b",
            sensor="test-sensor",
            modality="optical",
            result={"detected": False},
            confidence=0.90,
        )
    )

    engine = ExecutionEngine(registry)

    step = PlanStep(
        step_id="T3",
        task="verification",
        operation="verification",
    )

    result = engine.execute_step(step)

    assert result.success is False
    assert result.output["status"] == "abstain"
    assert result.output["verified"] is False
    assert len(result.output["conflicts"]) == 1


def test_executor_verification_checks_required_modalities():
    from src.executor.executor import ExecutionEngine
    from src.planner.evidence_planner import PlanStep
    from src.schemas.evidence import Evidence
    from src.evidence.registry import EvidenceRegistry

    registry = EvidenceRegistry()

    registry.add(
        Evidence(
            evidence_id="E_OPTICAL",
            source="test",
            task="sar_analysis",
            model="optical-model",
            sensor="test-sensor",
            modality="optical",
            result={"detected": True},
            confidence=0.90,
        )
    )

    engine = ExecutionEngine(registry)

    step = PlanStep(
        step_id="T2",
        task="verification",
        operation="verification",
        parameters={
            "required_modalities": ["optical", "sar"],
        },
    )

    result = engine.execute_step(step)

    assert result.success is False
    assert result.output["status"] == "abstain"
    assert "sar" in result.output["reasons"][0]


def test_executor_runs_temporal_analysis_specialist(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from src.executor.change_specialist import ChangeSpecialist
    from src.planner.evidence_planner import PlanStep

    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"

    before_data = np.zeros((10, 10), dtype=np.uint16)
    after_data = before_data.copy()
    after_data[2:6, 2:6] = 1000

    transform = from_origin(500000, 2000, 10, 10)

    for path, data in [(before, before_data), (after, after_data)]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            transform=transform,
        ) as dataset:
            dataset.write(data, 1)

    registry = EvidenceRegistry()
    engine = ExecutionEngine(registry)

    engine.register_specialist(ChangeSpecialist())

    step = PlanStep(
        step_id="T1",
        task="temporal_analysis",
        operation="temporal_analysis",
    )

    result = engine.execute_step(
        step,
        inputs=[str(before), str(after)],
    )

    assert result.success is True
    assert result.task == "temporal_analysis"
    assert len(result.evidence_ids) == 1

    evidence = registry.get(result.evidence_ids[0])

    assert evidence.task == "temporal_analysis"
    assert evidence.result["changed"] is True
    assert evidence.result["changed_pixels"] > 0


def test_executor_runs_gis_intersection_step():
    from shapely.geometry import box

    from src.planner.evidence_planner import PlanStep

    registry = EvidenceRegistry()

    registry.add(
        Evidence(
            evidence_id="E_BUILDING_GEOM",
            source="test-fixture",
            task="building_detection",
            model="test-building",
            modality="optical",
            geometry={
                "type": "geojson",
                "geometry": box(0, 0, 10, 10).__geo_interface__,
            },
            confidence=0.95,
        )
    )

    registry.add(
        Evidence(
            evidence_id="E_FLOOD_GEOM",
            source="test-fixture",
            task="flood_detection",
            model="test-flood",
            modality="sar",
            geometry={
                "type": "geojson",
                "geometry": box(5, 5, 15, 15).__geo_interface__,
            },
            confidence=0.90,
        )
    )

    engine = ExecutionEngine(registry)

    step = PlanStep(
        step_id="T3",
        task="gis_intersection",
        operation="intersection",
        parameters={},
        depends_on=["T1", "T2"],
    )

    result = engine.execute_step(step)

    assert result.success is True
    assert result.task == "gis_intersection"
    assert len(result.evidence_ids) == 1

    evidence = registry.get(result.evidence_ids[0])

    assert evidence.task == "gis_intersection"
    assert evidence.measurement["area_m2"] == 25.0
    assert evidence.provenance["source_evidence_ids"] == [
        "E_BUILDING_GEOM",
        "E_FLOOD_GEOM",
    ]


def test_executor_runs_gis_buffer_step():
    from shapely.geometry import box

    from src.planner.evidence_planner import PlanStep

    registry = EvidenceRegistry()

    registry.add(
        Evidence(
            evidence_id="E_REFERENCE",
            source="test-fixture",
            task="reference_detection",
            model="test-model",
            modality="optical",
            geometry={
                "type": "geojson",
                "geometry": box(0, 0, 10, 10).__geo_interface__,
            },
            confidence=0.92,
        )
    )

    engine = ExecutionEngine(registry)

    step = PlanStep(
        step_id="T2",
        task="gis_buffer",
        operation="buffer",
        parameters={"distance_m": 5},
        depends_on=["T1"],
    )

    result = engine.execute_step(step)

    assert result.success is True
    assert result.task == "gis_buffer"

    evidence = registry.get(result.evidence_ids[0])

    assert evidence.measurement["distance_m"] == 5.0
    assert evidence.measurement["area_m2"] > 100.0
    assert evidence.metadata["deterministic"] is True
