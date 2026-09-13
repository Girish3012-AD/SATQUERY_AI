from __future__ import annotations

from src.evidence.registry import EvidenceRegistry
from src.planner.evidence_planner import EvidencePlan, PlanStep
from src.verifier.georeason_verifier import GeoReasonVerifier

from .execution_result import ExecutionResult
from .specialist import Specialist


class ExecutionEngine:
    """
    Executes an EvidencePlan using registered specialist implementations
    and the GeoReason evidence verifier.

    Specialist inference produces Evidence.
    Verification evaluates the accumulated Evidence.
    """

    def __init__(
        self,
        evidence_registry: EvidenceRegistry,
        specialists: dict[str, Specialist] | None = None,
        verifier: GeoReasonVerifier | None = None,
    ) -> None:
        self.evidence_registry = evidence_registry
        self.specialists = specialists or {}
        self.verifier = verifier or GeoReasonVerifier()

    def register_specialist(self, specialist: Specialist) -> None:
        if specialist.capability in self.specialists:
            raise ValueError(
                f"Specialist already registered: {specialist.capability}"
            )

        self.specialists[specialist.capability] = specialist

    def execute_specialist(
        self,
        step: PlanStep,
        inputs: list[str] | None = None,
        specialist_override: Specialist | None = None,
        selected_model: str | None = None,
    ) -> ExecutionResult:
        """
        Execute a specialist.

        By default, specialists are selected by capability for backwards
        compatibility. The orchestrator may provide a model-bound specialist
        explicitly when the SensorAwareRouter has selected a concrete model.
        """
        specialist = specialist_override or self.specialists.get(step.task)

        if specialist is None:
            return ExecutionResult(
                success=False,
                step_id=step.step_id,
                task=step.task,
                message=(
                    f"No specialist implementation is registered "
                    f"for capability '{step.task}'."
                ),
            )

        evidence = specialist.infer(
            inputs or [],
            parameters=step.parameters,
        )

        # When orchestration selected a concrete model, make that selection
        # authoritative in the evidence record.
        if selected_model:
            evidence = evidence.model_copy(
                update={"model": selected_model}
            )

        self.evidence_registry.add(evidence)

        return ExecutionResult(
            success=True,
            step_id=step.step_id,
            task=step.task,
            output=evidence.result,
            evidence_ids=[evidence.evidence_id],
            message="Specialist execution completed.",
        )

    def execute_verification(
        self,
        step: PlanStep,
    ) -> ExecutionResult:
        """
        Verify all evidence accumulated in the EvidenceRegistry.

        Verification is performed after evidence-producing steps.
        """

        evidence = self.evidence_registry.all()

        expected_task = step.parameters.get("expected_task")

        if expected_task is not None and not isinstance(
            expected_task,
            str,
        ):
            return ExecutionResult(
                success=False,
                step_id=step.step_id,
                task=step.task,
                message="expected_task must be a string.",
            )

        required_modalities_value = step.parameters.get(
            "required_modalities",
            [],
        )

        if not isinstance(required_modalities_value, list):
            return ExecutionResult(
                success=False,
                step_id=step.step_id,
                task=step.task,
                message="required_modalities must be a list.",
            )

        verification = self.verifier.verify(
            evidence,
            expected_task=expected_task,
            required_modalities=required_modalities_value,
        )

        return ExecutionResult(
            success=verification.verified,
            step_id=step.step_id,
            task=step.task,
            output=verification.model_dump(),
            evidence_ids=verification.evidence_ids,
            message=(
                f"Verification status: {verification.status}. "
                f"Recommended action: "
                f"{verification.recommended_action or 'none'}."
            ),
        )

    def execute_step(
        self,
        step: PlanStep,
        inputs: list[str] | None = None,
        specialist_override: Specialist | None = None,
        selected_model: str | None = None,
    ) -> ExecutionResult:
        if step.operation == "specialist_inference":
            return self.execute_specialist(
                step,
                inputs,
                specialist_override=specialist_override,
                selected_model=selected_model,
            )

        if step.operation == "temporal_analysis":
            return self.execute_specialist(
                step,
                inputs,
                specialist_override=specialist_override,
                selected_model=selected_model,
            )

        if step.operation == "visual_question_answering":
            return self.execute_specialist(
                step,
                inputs,
                specialist_override=specialist_override,
                selected_model=selected_model,
            )

        if step.operation == "sar_analysis":
            return self.execute_specialist(
                step,
                inputs,
                specialist_override=specialist_override,
                selected_model=selected_model,
            )

        if step.operation == "verification":
            return self.execute_verification(step)

        return ExecutionResult(
            success=False,
            step_id=step.step_id,
            task=step.task,
            message=(
                f"Execution operation '{step.operation}' is not "
                f"implemented yet."
            ),
        )

    def execute(
        self,
        plan: EvidencePlan,
        inputs: list[str] | None = None,
        specialist_bindings: dict[str, Specialist] | None = None,
        selected_models: dict[str, str] | None = None,
    ) -> list[ExecutionResult]:
        """
        Execute a plan.

        ``specialist_bindings`` maps capability -> concrete specialist and is
        used by the orchestrator when model-aware routing is active.

        ``selected_models`` maps capability -> selected model name so the
        resulting Evidence records the authoritative routed model.
        """
        results: list[ExecutionResult] = []

        specialist_bindings = specialist_bindings or {}
        selected_models = selected_models or {}

        for step in plan.steps:
            result = self.execute_step(
                step,
                inputs,
                specialist_override=specialist_bindings.get(step.task),
                selected_model=selected_models.get(step.task),
            )

            results.append(result)

            if not result.success:
                break

        return results
