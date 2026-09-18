from pathlib import Path

import pytest

from src.controller.task_controller import TaskController
from src.evidence import EvidenceRegistry
from src.executor.executor import ExecutionEngine
from src.executor.vqa_specialist import (
    DEFAULT_ADAPTER_PATH,
    VqaSpecialist,
)
from src.orchestration.orchestrator import SATQueryOrchestrator
from src.planner.evidence_planner import EvidencePlanner
from src.registry import ModelRegistry, ModelSpec
from src.router.sensor_router import SensorAwareRouter


def find_vqa_image() -> str:
    candidates = [
        Path("data/samples/vqa_test.png"),
        Path(
            "data/remote_sensing/spacenet4/"
            "Pan-Sharpen_Atlanta_nadir53_catid_1030010003CD4300_"
            "743501_3721539.tif"
        ),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "No VQA test image found. Expected one of:\n"
        + "\n".join(str(path) for path in candidates)
    )


@pytest.mark.integration
def test_real_vqa_orchestration_end_to_end():
    image_path = find_vqa_image()

    adapter_path = Path(DEFAULT_ADAPTER_PATH)

    assert adapter_path.exists(), (
        f"VQA LoRA adapter does not exist: {adapter_path}"
    )

    assert (adapter_path / "adapter_config.json").exists(), (
        "VQA LoRA adapter_config.json is missing."
    )

    assert (adapter_path / "adapter_model.safetensors").exists(), (
        "VQA LoRA adapter_model.safetensors is missing."
    )

    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            name=VqaSpecialist.MODEL_NAME,
            capability=VqaSpecialist.CAPABILITY,
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            specialist_name="VqaSpecialist",
            checkpoint=DEFAULT_ADAPTER_PATH,
            version="2B-LoRA",
            metadata={
                "framework": "transformers",
                "execution": "local",
                "offline": True,
                "remote_sensing_adapted": True,
                "adapter_type": "PEFT_LORA",
                "adapter_path": DEFAULT_ADAPTER_PATH,
                "confidence_calibrated": False,
                "scientific_validation": False,
                "role": "evidence_grounded_visual_question_answering",
            },
        )
    )

    specialist = VqaSpecialist(
        adapter_path=DEFAULT_ADAPTER_PATH
    )

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

    query = "What is visible in this image?"

    print()
    print("=" * 60)
    print("STEP 2AY-Z-N : REAL VQA PRODUCTION E2E")
    print("=" * 60)
    print("Input:", image_path)
    print("Model:", VqaSpecialist.MODEL_NAME)
    print("Adapter:", DEFAULT_ADAPTER_PATH)
    print("Query:", query)

    result = orchestrator.run(
        query=query,
        inputs=[image_path],
    )

    print()
    print("=" * 60)
    print("ORCHESTRATION RESULT")
    print("=" * 60)
    print("success:", result.success)
    print("task_type:", result.task_type)
    print("selected_capabilities:", result.selected_capabilities)
    print("selected_models:", result.selected_models)
    print("executed_steps:", result.executed_steps)
    print("successful_steps:", result.successful_steps)
    print("failed_steps:", result.failed_steps)
    print("evidence_ids:", result.evidence_ids)
    print("verification:", result.verification)
    print("messages:", result.messages)

    # ---------------------------------------------------------
    # Controller / planner
    # ---------------------------------------------------------

    assert result.task_type == "vqa", (
        f"Expected vqa task type, got {result.task_type!r}"
    )

    # ---------------------------------------------------------
    # Automatic registry + router selection
    # ---------------------------------------------------------

    assert result.selected_models.get("vqa") == (
        VqaSpecialist.MODEL_NAME
    ), (
        "Orchestrator did not select the real Qwen VQA model."
    )

    assert result.selected_capabilities.get("vqa") == (
        VqaSpecialist.__name__
    ), (
        "Orchestrator did not bind the real VqaSpecialist."
    )

    # ---------------------------------------------------------
    # Specialist execution
    # ---------------------------------------------------------

    assert "T1" in result.successful_steps, (
        "Real VQA specialist execution did not succeed."
    )

    assert result.evidence_ids, (
        "Real VQA produced no evidence."
    )

    evidence_items = evidence_registry.all()

    assert evidence_items, (
        "Evidence registry is empty after real VQA execution."
    )

    vqa_evidence = next(
        (
            item
            for item in evidence_items
            if item.task == "vqa"
        ),
        None,
    )

    assert vqa_evidence is not None, (
        "No VQA evidence was found in the EvidenceRegistry."
    )

    # ---------------------------------------------------------
    # Evidence contract
    # ---------------------------------------------------------

    assert vqa_evidence.model == VqaSpecialist.MODEL_NAME
    assert vqa_evidence.task == "vqa"
    assert vqa_evidence.modality == "optical"

    print()
    print("=" * 60)
    print("REAL VQA EVIDENCE")
    print("=" * 60)
    print("evidence_id:", vqa_evidence.evidence_id)
    print("task:", vqa_evidence.task)
    print("model:", vqa_evidence.model)
    print("modality:", vqa_evidence.modality)
    print("confidence:", vqa_evidence.confidence)
    print("result:", vqa_evidence.result)
    print("provenance:", vqa_evidence.provenance)

    # ---------------------------------------------------------
    # Real LoRA / evidence-conditioning provenance
    # ---------------------------------------------------------

    provenance = vqa_evidence.provenance

    assert provenance.get("adapter_loaded") is True, (
        "Real VQA inference did not report adapter_loaded=True."
    )

    assert provenance.get("adapter_type") == "PEFT_LORA", (
        "Real VQA inference did not report PEFT_LORA."
    )

    assert provenance.get("remote_sensing_adapted") is True, (
        "Real VQA inference did not report remote-sensing adaptation."
    )

    assert provenance.get("evidence_provided") is False, (
        "Image-only production test unexpectedly received VQA evidence."
    )

    assert provenance.get("evidence_conditioned") is False, (
        "Image-only production test unexpectedly became evidence-conditioned."
    )

    # ---------------------------------------------------------
    # Confidence must remain honest
    # ---------------------------------------------------------

    assert vqa_evidence.confidence == pytest.approx(0.5), (
        "VQA confidence changed unexpectedly. "
        "The current confidence is intentionally uncalibrated."
    )

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------

    assert result.verification, (
        "Production VQA returned no verification result."
    )

    verification = result.verification

    print()
    print("=" * 60)
    print("REAL VQA VERIFICATION")
    print("=" * 60)
    print("status:", verification.get("status"))
    print("verified:", verification.get("verified"))
    print("confidence:", verification.get("confidence"))
    print("recommended_action:",
          verification.get("recommended_action"))
    print("vqa_claim_status:",
          verification.get("vqa_claim_status"))

    # Current VQA confidence is 0.5, below the 0.60 verifier
    # threshold. Therefore the specialist execution succeeds but
    # overall verification must remain low_confidence.
    assert verification.get("status") == "low_confidence", (
        "Expected current uncalibrated VQA evidence to remain "
        "low_confidence."
    )

    assert verification.get("verified") is False, (
        "Uncalibrated VQA confidence must not be marked verified."
    )

    assert verification.get("recommended_action") == (
        "request_additional_evidence"
    ), (
        "Expected low-confidence VQA verification to request "
        "additional evidence."
    )

    # No structured VQA consistency object is supplied in this
    # image-only production test, so claim status should remain None.
    assert verification.get("vqa_claim_status") is None, (
        "Image-only VQA should not invent a structured claim status."
    )

    # Overall orchestration is expected to fail verification, not
    # specialist execution.
    assert result.success is False, (
        "Expected overall result to remain unsuccessful because "
        "current VQA confidence is below the verifier threshold."
    )

    print()
    print("=" * 60)
    print("STEP 2AY-Z-N : REAL VQA PRODUCTION E2E PASS")
    print("=" * 60)
