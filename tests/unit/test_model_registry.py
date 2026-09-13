import pytest

from src.registry import ModelRegistry, ModelSpec


def test_register_and_get_model():
    registry = ModelRegistry()

    model = ModelSpec(
        name="test-vlm",
        capability="vqa",
        task_types=["vqa"],
        modalities=["optical"],
        status="PLANNED",
    )

    registry.register(model)

    assert registry.count() == 1
    assert registry.get("test-vlm").name == "test-vlm"


def test_duplicate_model_rejected():
    registry = ModelRegistry()

    model = ModelSpec(
        name="test-model",
        capability="vqa",
        status="PLANNED",
    )

    registry.register(model)

    with pytest.raises(ValueError):
        registry.register(model)


def test_invalid_status_rejected():
    registry = ModelRegistry()

    with pytest.raises(ValueError):
        registry.register(
            ModelSpec(
                name="bad-model",
                capability="vqa",
                status="FAKE_STATUS",
            )
        )


def test_find_available_models():
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

    registry.register(
        ModelSpec(
            name="available-vlm",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            checkpoint="/models/test",
        )
    )

    all_models = registry.find_by_capability("vqa")
    available = registry.find_by_capability(
        "vqa",
        available_only=True,
    )

    assert len(all_models) == 2
    assert len(available) == 1
    assert available[0].name == "available-vlm"


def test_find_by_task_and_modality():
    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            name="optical-vqa",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
        )
    )

    registry.register(
        ModelSpec(
            name="sar-vqa",
            capability="vqa",
            task_types=["vqa"],
            modalities=["sar"],
            status="AVAILABLE",
        )
    )

    result = registry.find(
        capability="vqa",
        task_type="vqa",
        modality="sar",
        available_only=True,
    )

    assert len(result) == 1
    assert result[0].name == "sar-vqa"


def test_missing_model_rejected():
    registry = ModelRegistry()

    with pytest.raises(KeyError):
        registry.get("DOES_NOT_EXIST")
