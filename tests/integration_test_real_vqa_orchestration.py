from pathlib import Path

from src.executor.vqa_specialist import VqaSpecialist
from src.orchestration.orchestrator import SATQueryOrchestrator
from src.registry import ModelRegistry, ModelSpec
from src.controller.task_controller import TaskController
from src.planner.evidence_planner import EvidencePlanner
from src.executor.executor import ExecutionEngine
from src.router.sensor_router import SensorAwareRouter
from src.evidence import EvidenceRegistry


def find_vqa_image() -> str:
    candidates = [
        Path("data/samples/vqa_test.png"),
        Path("data/remote_sensing/spacenet4/Pan-Sharpen_Atlanta_nadir53_catid_1030010003CD4300_743501_3721539.tif"),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "No VQA test image found. Expected one of:\n"
        + "\n".join(str(path) for path in candidates)
    )


def main() -> None:
    image_path = find_vqa_image()

    print("VQA input:", image_path)

    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            name="Qwen2-VL-2B-Instruct",
            capability="vqa",
            task_types=["vqa"],
            modalities=["optical"],
            status="AVAILABLE",
            specialist_name="VqaSpecialist",
            checkpoint="Qwen/Qwen2-VL-2B-Instruct",
            version="2B",
            metadata={
                "framework": "transformers",
                "execution": "local",
                "offline": True,
                "remote_sensing_adapted": False,
                "role": "visual_question_answering",
            },
        )
    )

    specialist = VqaSpecialist()

    controller = TaskController()
    planner = EvidencePlanner()
    evidence_registry = EvidenceRegistry()
    engine = ExecutionEngine(evidence_registry=evidence_registry)

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
    print("Query:", query)
    print("Registered model:", registry.get("Qwen2-VL-2B-Instruct"))
    print("Registered specialist:", specialist.__class__.__name__)

    result = orchestrator.run(
        query=query,
        inputs=[image_path],
    )

    print()
    print("==================================================")
    print("ORCHESTRATION RESULT")
    print("==================================================")

    print("success:", result.success)
    print("task_id:", result.task_id)
    print("task_type:", result.task_type)
    print("plan_id:", result.plan_id)
    print("selected_capabilities:", result.selected_capabilities)
    print("selected_models:", result.selected_models)
    print("executed_steps:", result.executed_steps)
    print("successful_steps:", result.successful_steps)
    print("failed_steps:", result.failed_steps)
    print("evidence_ids:", result.evidence_ids)
    print("verification:", result.verification)
    print("messages:", result.messages)

    assert result.selected_models.get("vqa") == "Qwen2-VL-2B-Instruct", (
        "Orchestrator did not select the real Qwen VQA model."
    )

    assert result.selected_capabilities.get("vqa") == "VqaSpecialist", (
        "Orchestrator did not bind the real VqaSpecialist."
    )

    # The specialist execution itself succeeded, but GeoReason is
    # expected to reject the current uncalibrated VQA confidence of 0.5.
    assert result.success is False, (
        "Expected verification to reject the uncalibrated VQA evidence."
    )

    assert "T1" in result.successful_steps, (
        "Real VQA specialist execution did not succeed."
    )

    assert "T2" in result.failed_steps, (
        "Verification step did not execute as expected."
    )

    assert result.verification.get("status") == "low_confidence", (
        "Expected GeoReason to mark the current VQA evidence as low_confidence."
    )

    assert result.verification.get("verified") is False, (
        "Low-confidence VQA evidence must not be marked verified."
    )

    assert (
        result.verification.get("recommended_action")
        == "request_additional_evidence"
    ), (
        "Expected verifier recommendation to request additional evidence."
    )

    assert result.evidence_ids, "Real VQA produced no evidence."

    evidence = evidence_registry.all()

    assert evidence, "Evidence registry is empty."

    vqa_evidence = evidence[0]

    print()
    print("==================================================")
    print("REAL EVIDENCE")
    print("==================================================")

    print("evidence_id:", vqa_evidence.evidence_id)
    print("task:", vqa_evidence.task)
    print("model:", vqa_evidence.model)
    print("modality:", vqa_evidence.modality)
    print("confidence:", vqa_evidence.confidence)
    print("result:", vqa_evidence.result)

    assert vqa_evidence.model == "Qwen2-VL-2B-Instruct", (
        "Evidence was not stamped with the routed model."
    )

    print()
    print("==================================================")
    print("PHASE 8C.1A - REAL VQA ORCHESTRATION: PASS")
    print("==================================================")


if __name__ == "__main__":
    main()
