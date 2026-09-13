import json
from pathlib import Path


AUDIT = Path(
    "outputs/checkpoints/"
    "qwen2vl_rs_vqa_evidence_grounded_dev/"
    "experiment_audit_9_4H_8B_9.json"
)


def test_evidence_grounded_adapter_audit_exists():
    assert AUDIT.exists()


def test_evidence_grounded_adapter_is_not_marked_rs_vlm():
    with AUDIT.open(encoding="utf-8") as f:
        data = json.load(f)

    assert data["integration_decision"]["remote_sensing_adapted"] is False
    assert data["integration_decision"]["production_registration"] is False


def test_evidence_perturbation_demonstrates_grounding():
    with AUDIT.open(encoding="utf-8") as f:
        data = json.load(f)

    assert data["evidence_grounding"]["prediction_change_rate"] == 1.0


def test_image_sensitivity_does_not_support_visual_reasoning_claim():
    with AUDIT.open(encoding="utf-8") as f:
        data = json.load(f)

    sensitivity = data["visual_sensitivity"]

    assert sensitivity["real_vs_blank_change_rate"] == 0.0
    assert sensitivity["real_vs_noise_change_rate"] == 0.0


def test_recommended_role_is_evidence_grounded_generator():
    with AUDIT.open(encoding="utf-8") as f:
        data = json.load(f)

    assert (
        data["integration_decision"]["recommended_role"]
        == "evidence_grounded_answer_generator"
    )
