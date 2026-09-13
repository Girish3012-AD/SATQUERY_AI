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
    ) -> ExecutionResult:
        specialist = self.specialists.get(step.task)

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
    ) -> ExecutionResult:
        if step.operation == "specialist_inference":
            return self.execute_specialist(step, inputs)

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
    ) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []

        for step in plan.steps:
            result = self.execute_step(step, inputs)

            results.append(result)

            if not result.success:
                break

        return results
