import pytest

from src.evidence import EvidenceRegistry
from src.executor import ExecutionEngine, Specialist
from src.orchestration import SATQueryOrchestrator
from src.schemas import Evidence


class FakeVqaSpecialist(Specialist):

    @property
    def capability(self) -> str:
        return "vqa"

    def infer(self, inputs, parameters=None):
        return Evidence(
            evidence_id="E-VQA-001",
            source="test-fixture",
            task="vqa",
            model="FakeVqaSpecialist",
            sensor="test-sensor",
            modality="optical",
            result={
                "answer": "A building is visible."
            },
            confidence=0.95,
            provenance={
                "test_only": True,
                "inputs": inputs,
            },
        )


class FakeChangeSpecialist(Specialist):

    @property
    def capability(self) -> str:
        return "temporal_analysis"

    def infer(self, inputs, parameters=None):
        return Evidence(
            evidence_id="E-CHANGE-001",
            source="test-fixture",
            task="temporal_analysis",
            model="FakeChangeSpecialist",
            sensor="test-sensor",
            modality="optical",
            t1_timestamp="2024-01-01T00:00:00Z",
            t2_timestamp="2024-06-01T00:00:00Z",
            geometry={
                "type": "Polygon",
                "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
            },
            result={
                "changed": True,
                "changed_pixels": 100,
            },
            confidence=0.95,
            provenance={
                "test_only": True,
                "inputs": inputs,
            },
        )


def test_orchestrator_builds_task_and_plan():

    orchestrator = SATQueryOrchestrator(
        specialists=[FakeVqaSpecialist()]
    )

    task = orchestrator.build_task_spec(
        query="What is shown in this image?",
        input_count=1,
    )

    plan = orchestrator.build_plan(task)

    assert task.task_type == "vqa"
    assert task.required_capabilities == ["vqa"]

    assert plan.plan_id == f"PLAN-{task.task_id}"
    assert plan.step_ids() == ["T1", "T2"]

    assert plan.get_step("T1").task == "vqa"
    assert plan.get_step("T2").operation == "verification"


def test_orchestrator_runs_vqa_end_to_end():

    orchestrator = SATQueryOrchestrator(
        specialists=[FakeVqaSpecialist()]
    )

    result = orchestrator.run(
        query="What is shown in this image?",
        inputs=["test.tif"],
    )

    assert result.task_type == "vqa"
    assert result.success is True

    assert result.executed_steps == ["T1", "T2"]
    assert result.successful_steps == ["T1", "T2"]
    assert result.failed_steps == []

    assert result.evidence_ids == ["E-VQA-001"]

    assert result.verification["status"] == "verified"
    assert result.verification["verified"] is True
    assert result.verification["confidence"] == 0.95

    assert result.selected_capabilities == {
        "vqa": "FakeVqaSpecialist"
    }

    assert result.status == "verified"


def test_orchestrator_runs_temporal_analysis_end_to_end():

    orchestrator = SATQueryOrchestrator(
        specialists=[FakeChangeSpecialist()]
    )

    result = orchestrator.run(
        query="Detect changes between the two images.",
        inputs=[
            "before.tif",
            "after.tif",
        ],
    )

    assert result.task_type == "temporal_analysis"
    assert result.success is True

    assert result.executed_steps == ["T1", "T2"]
    assert result.successful_steps == ["T1", "T2"]

    assert result.evidence_ids == ["E-CHANGE-001"]

    assert result.verification["status"] == "verified"
    assert result.verification["verified"] is True
    assert result.status == "verified"


def test_orchestrator_detects_missing_specialist():
    # Explicitly provide no specialists so this test exercises the
    # missing-specialist routing path.
    orchestrator = SATQueryOrchestrator(
        specialists=[]
    )

    result = orchestrator.run(
        query="What is shown in this image?",
        inputs=["test.tif"],
    )

    assert result.success is False
    assert result.executed_steps == []
    assert result.evidence_ids == []
    assert result.status == "failed"

    assert "Required specialist implementation(s) unavailable." in (
        result.messages
    )

    assert any(
        "vqa" in message
        for message in result.messages
    )

def test_orchestrator_rejects_duplicate_specialist():

    orchestrator = SATQueryOrchestrator(
        specialists=[FakeVqaSpecialist()]
    )

    with pytest.raises(ValueError):
        orchestrator.register_specialist(
            FakeVqaSpecialist()
        )


def test_orchestrator_exposes_registered_specialists():

    specialist = FakeVqaSpecialist()

    orchestrator = SATQueryOrchestrator(
        specialists=[specialist]
    )

    specialists = orchestrator.specialists

    assert "vqa" in specialists
    assert specialists["vqa"] is specialist


def test_orchestrator_collects_multiple_evidence_ids():

    class MultiEvidenceSpecialist(Specialist):

        @property
        def capability(self) -> str:
            return "building_detection"

        def infer(self, inputs, parameters=None):
            return Evidence(
                evidence_id="E-BUILDING-001",
                source="test-fixture",
                task="building_detection",
                model="MultiEvidenceSpecialist",
                sensor="test-sensor",
                modality="optical",
                result={"detected": True},
                confidence=0.90,
                provenance={"test_only": True},
            )

    orchestrator = SATQueryOrchestrator(
        specialists=[MultiEvidenceSpecialist()]
    )

    result = orchestrator.run(
        query="Find buildings.",
        inputs=["test.tif"],
    )

    assert result.success is True
    assert result.evidence_ids == ["E-BUILDING-001"]
    assert result.status == "verified"


def test_orchestrator_execute_alias_matches_run():

    orchestrator = SATQueryOrchestrator(
        specialists=[FakeVqaSpecialist()]
    )

    result_run = orchestrator.run(
        query="What is shown in this image?",
        inputs=["test.tif"],
    )

    orchestrator_2 = SATQueryOrchestrator(
        specialists=[FakeVqaSpecialist()]
    )

    result_execute = orchestrator_2.execute(
        query="What is shown in this image?",
        inputs=["test.tif"],
    )

    assert result_run.query == result_execute.query
    assert result_run.task_type == result_execute.task_type
    assert result_run.evidence_ids == result_execute.evidence_ids
    assert result_run.status == result_execute.status
