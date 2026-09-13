import pytest

from src.registry import ModelRegistry, ModelSpec
from src.router import SensorAwareRouter
from src.schemas import TaskSpec


def make_task(
    task_type="vqa",
    capabilities=None,
    modalities=None,
):
    return TaskSpec(
        task_id="TASK-001",
        query="Test query",
        task_type=task_type,
        required_capabilities=capabilities or ["vqa"],
        required_modalities=modalities or [],
        input_count=1,
    )


def test_routes_to_available_model():
    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            name="test-vlm",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            checkpoint="/models/test-vlm",
        )
    )

    router = SensorAwareRouter(registry)

    result = router.route(
        make_task(),
        capability="vqa",
        modality="optical",
    )

    assert result.routed is True
    assert result.model is not None
    assert result.model.name == "test-vlm"


def test_does_not_route_planned_model():
    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            name="planned-vlm",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="PLANNED",
        )
    )

    router = SensorAwareRouter(registry)

    result = router.route(
        make_task(),
        capability="vqa",
        modality="optical",
    )

    assert result.routed is False
    assert result.model is None
    assert "No AVAILABLE model" in result.reason


def test_rejects_wrong_modality():
    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            name="sar-model",
            capability="vqa",
            task_types=["vqa"],
            modalities=["sar"],
            status="AVAILABLE",
        )
    )

    router = SensorAwareRouter(registry)

    result = router.route(
        make_task(),
        capability="vqa",
        modality="optical",
    )

    assert result.routed is False
    assert result.model is None


def test_rejects_wrong_task_type():
    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            name="vqa-model",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
        )
    )

    router = SensorAwareRouter(registry)

    task = make_task(task_type="specialized_analysis")

    result = router.route(
        task,
        capability="vqa",
        modality="optical",
    )

    assert result.routed is False
    assert result.model is None


def test_selects_first_available_and_reports_alternatives():
    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            name="model-a",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
        )
    )

    registry.register(
        ModelSpec(
            name="model-b",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
        )
    )

    router = SensorAwareRouter(registry)

    result = router.route(
        make_task(),
        capability="vqa",
        modality="optical",
    )

    assert result.routed is True
    assert result.model.name == "model-a"
    assert result.alternatives == ["model-b"]
