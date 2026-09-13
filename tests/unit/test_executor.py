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
