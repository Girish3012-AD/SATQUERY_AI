from pathlib import Path

import pytest

from src.executor.vqa_specialist import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL_PATH,
    VqaSpecialist,
)


ADAPTER = Path(DEFAULT_ADAPTER_PATH)


def test_default_vqa_specialist_remains_base_only():
    specialist = VqaSpecialist()

    assert specialist.model_path == DEFAULT_MODEL_PATH
    assert specialist.adapter_path is None
    assert specialist.capability == "vqa"


def test_adapter_path_is_explicitly_configured():
    specialist = VqaSpecialist(
        adapter_path=DEFAULT_ADAPTER_PATH
    )

    assert specialist.adapter_path == str(ADAPTER.resolve())
    assert Path(specialist.adapter_path).exists()


def test_adapter_artifacts_exist():
    assert ADAPTER.exists()
    assert (ADAPTER / "adapter_config.json").is_file()
    assert (ADAPTER / "adapter_model.safetensors").is_file()


def test_invalid_adapter_path_is_rejected_on_load():
    specialist = VqaSpecialist(
        adapter_path="/tmp/satquery-nonexistent-vqa-adapter"
    )

    with pytest.raises(FileNotFoundError, match="adapter path"):
        specialist._load_model()

    specialist.unload()


def test_missing_adapter_config_is_rejected(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    (adapter / "adapter_model.safetensors").write_bytes(
        b"placeholder"
    )

    specialist = VqaSpecialist(
        adapter_path=str(adapter)
    )

    with pytest.raises(
        FileNotFoundError,
        match="adapter configuration",
    ):
        specialist._load_model()

    specialist.unload()


def test_missing_adapter_weights_are_rejected(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    (adapter / "adapter_config.json").write_text("{}")

    specialist = VqaSpecialist(
        adapter_path=str(adapter)
    )

    with pytest.raises(
        FileNotFoundError,
        match="adapter weights",
    ):
        specialist._load_model()

    specialist.unload()


def test_adapter_constant_points_to_expected_development_checkpoint():
    assert (
        "qwen2vl_rs_vqa_evidence_grounded_dev"
        in DEFAULT_ADAPTER_PATH
    )
