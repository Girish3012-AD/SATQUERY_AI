from pathlib import Path

from src.executor.executor import ExecutionEngine
from src.executor.specialist import Specialist
from src.planner.evidence_planner import EvidencePlanner
from src.registry.model_registry import ModelRegistry
from src.registry.model_spec import ModelSpec
from src.router.sensor_router import SensorAwareRouter
from src.schemas.evidence import Evidence
from src.orchestration.orchestrator import SATQueryOrchestrator


class FakeSpecialist(Specialist):
    def __init__(self, capability: str):
        self._capability = capability

    @property
    def capability(self) -> str:
        return self._capability

    def infer(self, inputs, parameters=None):
        return Evidence(
            evidence_id=f"fake-{self._capability}",
            source="unit-test",
            task=self._capability,
            model=f"Fake-{self._capability}",
            sensor="test",
            modality="optical",
            confidence=0.9,
            result={"answer": "synthetic test evidence"},
            provenance={"test": True},
        )


def make_registry(*models):
    registry = ModelRegistry()
    for model in models:
        registry.register(model)
    return registry


def test_vqa_routes_to_available_optical_vlm():
    registry = make_registry(
        ModelSpec(
            name="fake-optical-vlm",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            checkpoint="fake/vlm",
            version="test",
        )
    )

    router = SensorAwareRouter(registry)
    orchestrator = SATQueryOrchestrator(
        specialists=[FakeSpecialist("vqa")],
        registry=registry,
        router=router,
    )

    result = orchestrator.run(
        query="What is shown in this optical image?",
        inputs=["test.tif"],
    )

    assert result.success is True
    assert result.selected_capabilities["vqa"] == "FakeSpecialist"
    assert result.selected_models["vqa"] == "fake-optical-vlm"


def test_sar_task_routes_to_available_sar_model():
    registry = make_registry(
        ModelSpec(
            name="fake-sar-model",
            capability="sar_analysis",
            task_types=["specialized_analysis"],
            modalities=["sar"],
            status="AVAILABLE",
            checkpoint="fake/sar",
            version="test",
        ),
        ModelSpec(
            name="fake-flood-model",
            capability="flood_detection",
            task_types=["specialized_analysis"],
            modalities=["sar"],
            status="AVAILABLE",
            checkpoint="fake/flood",
            version="test",
        ),
    )

    router = SensorAwareRouter(registry)

    orchestrator = SATQueryOrchestrator(
        specialists=[
            FakeSpecialist("sar_analysis"),
            FakeSpecialist("flood_detection"),
        ],
        registry=registry,
        router=router,
    )

    result = orchestrator.run(
        query="Analyze this SAR image for flooded areas.",
        inputs=["test.tif"],
    )

    assert result.success is True
    assert result.selected_capabilities["sar_analysis"] == "FakeSpecialist"
    assert result.selected_models["sar_analysis"] == "fake-sar-model"


def test_planned_model_is_not_routed():
    registry = make_registry(
        ModelSpec(
            name="planned-sar-model",
            capability="sar_analysis",
            task_types=["sar_analysis"],
            modalities=["sar"],
            status="PLANNED",
            checkpoint="fake/sar",
            version="test",
        )
    )

    router = SensorAwareRouter(registry)

    orchestrator = SATQueryOrchestrator(
        specialists=[FakeSpecialist("sar_analysis")],
        registry=registry,
        router=router,
    )

    result = orchestrator.run(
        query="Analyze this SAR image.",
        inputs=["test.tif"],
    )

    assert result.success is False
    assert result.executed_steps == []
    assert result.selected_models == {}
    assert any(
        "routing failed" in message.lower()
        or "available" in message.lower()
        for message in result.messages
    )


def test_wrong_modality_is_not_routed():
    registry = make_registry(
        ModelSpec(
            name="optical-only-vlm",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            checkpoint="fake/vlm",
            version="test",
        )
    )

    router = SensorAwareRouter(registry)

    orchestrator = SATQueryOrchestrator(
        specialists=[FakeSpecialist("vqa")],
        registry=registry,
        router=router,
    )

    # Directly verify that a SAR requirement cannot use the optical model.
    task_spec = orchestrator.build_task_spec(
        query="Analyze this SAR image.",
        input_count=1,
    )

    route = router.route(
        task_spec=task_spec,
        capability="vqa",
        modality="sar",
    )

    assert route.routed is False
    assert route.model is None
    assert route.alternatives == []


def test_multiple_available_models_preserve_alternatives():
    registry = make_registry(
        ModelSpec(
            name="vlm-primary",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            checkpoint="fake/primary",
            version="test",
        ),
        ModelSpec(
            name="vlm-secondary",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            checkpoint="fake/secondary",
            version="test",
        ),
    )

    router = SensorAwareRouter(registry)

    orchestrator = SATQueryOrchestrator(
        specialists=[FakeSpecialist("vqa")],
        registry=registry,
        router=router,
    )

    task_spec = orchestrator.build_task_spec(
        query="What is shown in this optical image?",
        input_count=1,
    )

    route = router.route(
        task_spec=task_spec,
        capability="vqa",
        modality="optical",
    )

    assert route.routed is True
    assert route.model is not None
    assert route.model.name == "vlm-primary"
    assert "vlm-secondary" in route.alternatives


def test_orchestrator_records_selected_model_for_auditing():
    registry = make_registry(
        ModelSpec(
            name="auditable-vlm",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            checkpoint="fake/auditable",
            version="test",
            metadata={"source": "unit-test"},
        )
    )

    orchestrator = SATQueryOrchestrator(
        specialists=[FakeSpecialist("vqa")],
        registry=registry,
        router=SensorAwareRouter(registry),
    )

    result = orchestrator.run(
        query="What is shown in this image?",
        inputs=["test.tif"],
    )

    assert result.success is True
    assert result.selected_models == {"vqa": "auditable-vlm"}
    assert result.selected_capabilities == {"vqa": "FakeSpecialist"}


class BoundVqaSpecialistA(Specialist):
    @property
    def capability(self) -> str:
        return "vqa"

    def infer(self, inputs, parameters=None):
        return Evidence(
            evidence_id="bound-vqa-a",
            source="unit-test",
            task="vqa",
            model="INTERNAL-MODEL-A",
            sensor="test",
            modality="optical",
            confidence=0.95,
            result={"answer": "ANSWER-FROM-MODEL-A"},
            provenance={"test": True},
        )


class BoundVqaSpecialistB(Specialist):
    @property
    def capability(self) -> str:
        return "vqa"

    def infer(self, inputs, parameters=None):
        return Evidence(
            evidence_id="bound-vqa-b",
            source="unit-test",
            task="vqa",
            model="INTERNAL-MODEL-B",
            sensor="test",
            modality="optical",
            confidence=0.95,
            result={"answer": "ANSWER-FROM-MODEL-B"},
            provenance={"test": True},
        )


def test_routed_model_is_bound_to_actual_specialist():
    registry = make_registry(
        ModelSpec(
            name="model-a",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            specialist_name="BoundVqaSpecialistA",
            checkpoint="fake/model-a",
            version="test",
        ),
        ModelSpec(
            name="model-b",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            specialist_name="BoundVqaSpecialistB",
            checkpoint="fake/model-b",
            version="test",
        ),
    )

    orchestrator = SATQueryOrchestrator(
        specialists=[
            BoundVqaSpecialistA(),
            BoundVqaSpecialistB(),
        ],
        registry=registry,
        router=SensorAwareRouter(registry),
    )

    result = orchestrator.run(
        query="What is shown in this optical image?",
        inputs=["test.tif"],
    )

    assert result.success is True

    # Router selected Model A.
    assert result.selected_models["vqa"] == "model-a"

    # Model A is actually bound to Specialist A.
    assert result.selected_capabilities["vqa"] == "BoundVqaSpecialistA"

    # Specialist A's result was actually executed.
    assert result.evidence_ids == ["bound-vqa-a"]

    evidence = orchestrator.evidence_registry.get("bound-vqa-a")

    assert evidence is not None
    assert evidence.model == "model-a"
    assert evidence.result["answer"] == "ANSWER-FROM-MODEL-A"


def test_routed_second_model_executes_its_bound_specialist():
    registry = make_registry(
        ModelSpec(
            name="model-b",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            specialist_name="BoundVqaSpecialistB",
            checkpoint="fake/model-b",
            version="test",
        ),
        ModelSpec(
            name="model-a",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            specialist_name="BoundVqaSpecialistA",
            checkpoint="fake/model-a",
            version="test",
        ),
    )

    orchestrator = SATQueryOrchestrator(
        specialists=[
            BoundVqaSpecialistA(),
            BoundVqaSpecialistB(),
        ],
        registry=registry,
        router=SensorAwareRouter(registry),
    )

    result = orchestrator.run(
        query="What is shown in this optical image?",
        inputs=["test.tif"],
    )

    assert result.success is True

    # Registry ordering causes Model B to be selected.
    assert result.selected_models["vqa"] == "model-b"

    # The selected model is now bound to Specialist B.
    assert result.selected_capabilities["vqa"] == "BoundVqaSpecialistB"

    # Specialist B actually executed.
    assert result.evidence_ids == ["bound-vqa-b"]

    evidence = orchestrator.evidence_registry.get("bound-vqa-b")

    assert evidence is not None
    assert evidence.model == "model-b"
    assert evidence.result["answer"] == "ANSWER-FROM-MODEL-B"
