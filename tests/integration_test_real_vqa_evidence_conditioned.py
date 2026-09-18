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
from src.verifier.georeason_verifier import GeoReasonVerifier


def find_evidence_image() -> str:
    candidates = [
        Path(
            "data/remote_sensing/spacenet4/patches/"
            "Atlanta_743501_3721539/images/"
            "patch_00188.npy"
        ),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find patch_00188.npy. Expected one of:\n"
        + "\n".join(str(path) for path in candidates)
    )


@pytest.mark.integration
def test_real_evidence_conditioned_vqa_orchestration():
    image_path = find_evidence_image()

    adapter_path = Path(DEFAULT_ADAPTER_PATH)

    assert adapter_path.exists()
    assert (adapter_path / "adapter_config.json").exists()
    assert (adapter_path / "adapter_model.safetensors").exists()

    # This is the real structured evidence from the audited
    # rs_vqa_evidence_grounded validation example 00188.
    evidence_text = (
        "Remote-sensing building-mask evidence: "
        "928 of 4096 pixels are classified as building pixels "
        "(22.66% coverage). "
        "The derived density category is "
        "'high building density'. "
        "The derived scene category is "
        "'mostly built-up'."
    )

    query = (
        "Using the provided remote-sensing evidence, "
        "summarize the building information in this image."
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

    evidence_registry = EvidenceRegistry()

    verifier = GeoReasonVerifier(
        minimum_confidence=0.60,
        minimum_evidence=1,
    )

    engine = ExecutionEngine(
        evidence_registry=evidence_registry,
        verifier=verifier,
    )

    orchestrator = SATQueryOrchestrator(
        controller=TaskController(),
        planner=EvidencePlanner(),
        evidence_registry=evidence_registry,
        engine=engine,
        specialists=[specialist],
        registry=registry,
        router=SensorAwareRouter(registry),
    )

    print()
    print("=" * 70)
    print("STEP 2AY-Z-O : REAL EVIDENCE-CONDITIONED VQA E2E")
    print("=" * 70)
    print("Image:", image_path)
    print("Model:", VqaSpecialist.MODEL_NAME)
    print("Adapter:", DEFAULT_ADAPTER_PATH)
    print("Query:", query)
    print("Evidence:", evidence_text)

    result = orchestrator.run(
        query=query,
        inputs=[image_path],
        parameters={
            "evidence": evidence_text,
            "vqa_structured_evidence": {
                "building_percentage": 22.66,
                "density": "high building density",
                "coverage": "high building coverage",
                "scene": "mostly built-up",
            },
            "max_new_tokens": 128,
        },
    )

    print()
    print("=" * 70)
    print("ORCHESTRATION RESULT")
    print("=" * 70)
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
    # 1. Controller / planner
    # ---------------------------------------------------------

    assert result.task_type == "vqa"

    # ---------------------------------------------------------
    # 2. Automatic routing
    # ---------------------------------------------------------

    assert result.selected_models.get("vqa") == (
        VqaSpecialist.MODEL_NAME
    )

    assert result.selected_capabilities.get("vqa") == (
        VqaSpecialist.__name__
    )

    # ---------------------------------------------------------
    # 3. Specialist execution
    # ---------------------------------------------------------

    assert "T1" in result.successful_steps
    assert result.evidence_ids

    vqa_evidence = next(
        (
            item
            for item in evidence_registry.all()
            if item.task == "vqa"
        ),
        None,
    )

    assert vqa_evidence is not None

    # ---------------------------------------------------------
    # 4. Evidence contract
    # ---------------------------------------------------------

    assert vqa_evidence.model == VqaSpecialist.MODEL_NAME
    assert vqa_evidence.task == "vqa"
    assert vqa_evidence.modality == "optical"

    # ---------------------------------------------------------
    # 5. LoRA integrity
    # ---------------------------------------------------------

    provenance = vqa_evidence.provenance

    assert provenance.get("adapter_loaded") is True
    assert provenance.get("adapter_type") == "PEFT_LORA"
    assert provenance.get("remote_sensing_adapted") is True

    # ---------------------------------------------------------
    # 6. Evidence conditioning MUST be active
    # ---------------------------------------------------------

    assert provenance.get("evidence_provided") is True
    assert provenance.get("evidence_conditioned") is True

    # The provenance should also record the temporary model image,
    # proving the .npy -> RGB PNG ingestion boundary was used.
    assert provenance.get("image_input_conversion") == (
        "npy_to_rgb_png"
    )

    # ---------------------------------------------------------
    # 7. Model must actually produce an answer
    # ---------------------------------------------------------

    answer = vqa_evidence.result.get("answer")

    assert isinstance(answer, str)
    assert answer.strip()

    print()
    print("=" * 70)
    print("VQA EVIDENCE")
    print("=" * 70)
    print("Evidence ID:", vqa_evidence.evidence_id)
    print("Answer:", answer)
    print("Confidence:", vqa_evidence.confidence)
    print("Evidence conditioned:",
          provenance.get("evidence_conditioned"))
    print("Evidence provided:",
          provenance.get("evidence_provided"))
    print("Adapter loaded:",
          provenance.get("adapter_loaded"))

    # ---------------------------------------------------------
    # 8. Current confidence policy remains unchanged
    # ---------------------------------------------------------

    assert vqa_evidence.confidence == pytest.approx(0.5)

    verification = result.verification

    assert verification

    # The real VQA specialist is still uncalibrated at 0.5.
    # Evidence conditioning must NOT artificially increase confidence.
    assert verification.get("confidence") == pytest.approx(0.5)

    assert verification.get("status") == "low_confidence"
    assert verification.get("verified") is False

    # ---------------------------------------------------------
    # 9. VQA structured claim verification
    # ---------------------------------------------------------

    # The supplied evidence and the model answer are now evaluated
    # by the structured VQA consistency layer before final verification.
    #
    # The claim is supported by the supplied remote-sensing evidence.
    assert verification.get("vqa_claim_status") == "SUPPORTED"

    # This does NOT override the model confidence policy.
    # The VQA specialist remains uncalibrated at confidence 0.5,
    # therefore the overall verification status remains low_confidence.
    assert verification.get("status") == "low_confidence"
    assert verification.get("verified") is False

    print()
    print("=" * 70)
    print("STEP 2AY-Z-O : EVIDENCE CONDITIONING TRANSPORT PASS")
    print("=" * 70)
