from typing import Any

from src.evidence import EvidenceRegistry
from src.planner import EvidencePlan, PlanStep
from src.schemas import Evidence

from .execution_result import ExecutionResult
from .specialist import Specialist


class ExecutionEngine:
    """
    Executes an EvidencePlan using registered specialist implementations.

    Specialist inference is delegated to real Specialist implementations.
    Deterministic GIS operations can be added independently.
    """

    def __init__(
        self,
        evidence_registry: EvidenceRegistry,
        specialists: dict[str, Specialist] | None = None,
    ) -> None:
        self.evidence_registry = evidence_registry
        self.specialists = specialists or {}

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

    def execute_step(
        self,
        step: PlanStep,
        inputs: list[str] | None = None,
    ) -> ExecutionResult:
        if step.operation == "specialist_inference":
            return self.execute_specialist(step, inputs)

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
