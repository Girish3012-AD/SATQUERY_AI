"""Focused tests for the VQA E2E path."""
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.controller.task_controller import TaskController
from src.executor.vqa_specialist import VqaSpecialist, DEFAULT_ADAPTER_PATH
from src.schemas.evidence import Evidence
from src.verifier.georeason_verifier import GeoReasonVerifier


# ── A/B: TaskController routes VQA query correctly ──────────
def test_vqa_query_routing():
    controller = TaskController()
    spec = controller.build_task_spec(
        query="Describe the land-cover and major objects visible in this image.",
        input_count=1,
    )
    assert spec.task_type == "vqa"
    assert "vqa" in spec.required_capabilities


def test_vqa_describe_query_routing():
    controller = TaskController()
    spec = controller.build_task_spec(
        query="What objects are visible in this satellite image?",
        input_count=1,
    )
    assert spec.task_type == "vqa"


# ── C: Missing image rejection ─────────────────────────────
def test_vqa_rejects_missing_image():
    spec = VqaSpecialist(model_path="dummy", adapter_path=None)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        spec.infer(
            inputs=["non_existent_image.png"],
            parameters={"query": "What is visible?"},
        )


def test_vqa_rejects_empty_inputs():
    spec = VqaSpecialist(model_path="dummy", adapter_path=None)
    with pytest.raises(ValueError, match="at least one image"):
        spec.infer(inputs=[], parameters={"query": "What is visible?"})


# ── D: Missing model rejection ─────────────────────────────
def test_vqa_rejects_missing_model(tmp_path):
    img = tmp_path / "img.png"
    from PIL import Image
    Image.new("RGB", (10, 10)).save(img)
    spec = VqaSpecialist(model_path="non_existent_model_dir_xyz", adapter_path=None)
    with pytest.raises(Exception):
        # Should fail when trying to load from nonexistent path
        spec.infer(inputs=[str(img)], parameters={"query": "What is visible?"})


# ── E: Missing adapter rejection ───────────────────────────
def test_vqa_rejects_missing_adapter(tmp_path):
    img = tmp_path / "img.png"
    from PIL import Image
    Image.new("RGB", (10, 10)).save(img)
    spec = VqaSpecialist(
        model_path="dummy",
        adapter_path=str(tmp_path / "nonexistent_adapter"),
    )
    # Should fail during _load_model
    with pytest.raises(Exception):
        spec.infer(inputs=[str(img)], parameters={"query": "What is visible?"})


# ── F/G: Evidence generation with provenance ────────────────
def test_vqa_evidence_has_required_fields():
    """Verify that VQA evidence schema includes all required fields."""
    ev = Evidence(
        evidence_id="VQA_test_abc123",
        source="vqa_specialist",
        task="vqa",
        model="Qwen2-VL-2B-Instruct",
        sensor="Sentinel-2",
        modality="optical",
        timestamp="2024-10-13",
        geometry=None,
        measurement={},
        result={
            "question": "Describe land-cover",
            "answer": "The image shows mountainous terrain with vegetation.",
        },
        confidence=0.5,
        provenance={
            "inference_type": "image_conditioned_vqa",
            "model_path": "Qwen/Qwen2-VL-2B-Instruct",
            "adapter_loaded": True,
            "adapter_type": "PEFT_LORA",
            "confidence_method": "uncalibrated_default; not a model probability",
        },
        metadata={"max_new_tokens": 256},
    )
    assert ev.task == "vqa"
    assert ev.model == "Qwen2-VL-2B-Instruct"
    assert "answer" in ev.result
    assert "inference_type" in ev.provenance
    assert "confidence_method" in ev.provenance
    assert ev.provenance["adapter_loaded"] is True


# ── H: Verifier handles VQA evidence ───────────────────────
def test_verifier_accepts_valid_vqa_evidence():
    ev = Evidence(
        evidence_id="VQA_test_valid",
        source="vqa_specialist",
        task="vqa",
        model="Qwen2-VL-2B-Instruct",
        sensor="Sentinel-2",
        modality="optical",
        timestamp="2024-10-13",
        geometry=None,
        measurement={},
        result={"answer": "Forest and mountains"},
        confidence=0.5,
        provenance={
            "inference_type": "image_conditioned_vqa",
            "confidence_method": "uncalibrated_default",
        },
        metadata={},
    )
    verifier = GeoReasonVerifier(minimum_confidence=0.3)
    res = verifier.verify([ev], expected_task="vqa", required_modalities=["optical"])
    assert res.verified is True
    assert res.status == "verified"


def test_verifier_rejects_low_confidence_vqa():
    ev = Evidence(
        evidence_id="VQA_test_low",
        source="vqa_specialist",
        task="vqa",
        model="Qwen2-VL-2B-Instruct",
        sensor="Sentinel-2",
        modality="optical",
        timestamp="2024-10-13",
        geometry=None,
        measurement={},
        result={"answer": "Forest"},
        confidence=0.1,
        provenance={},
        metadata={},
    )
    verifier = GeoReasonVerifier(minimum_confidence=0.5)
    res = verifier.verify([ev], expected_task="vqa")
    assert res.verified is False


# ── I: Evidence ID determinism ─────────────────────────────
def test_vqa_evidence_id_deterministic():
    id1 = VqaSpecialist._make_evidence_id("image.tif", "Describe land-cover")
    id2 = VqaSpecialist._make_evidence_id("image.tif", "Describe land-cover")
    assert id1 == id2
    assert id1.startswith("VQA_")


def test_vqa_evidence_id_varies_with_query():
    id1 = VqaSpecialist._make_evidence_id("image.tif", "Query A")
    id2 = VqaSpecialist._make_evidence_id("image.tif", "Query B")
    assert id1 != id2
