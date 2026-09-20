from __future__ import annotations

from src.evidence.registry import EvidenceRegistry
from src.planner.evidence_planner import EvidencePlan, PlanStep
from src.verifier.georeason_verifier import GeoReasonVerifier
from src.verifier.vqa_consistency import evaluate_vqa_consistency

from .execution_result import ExecutionResult
from .gis_executor import GISEvidenceExecutor
from .multimodal_executor import MultimodalEvidenceExecutor
from .change_geospatializer import ChangeGeospatializer
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
        self.gis_executor = GISEvidenceExecutor(evidence_registry)
        self.multimodal_executor = MultimodalEvidenceExecutor(
            evidence_registry
        )
        self.change_geospatializer = ChangeGeospatializer()

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

        # Temporal specialists require an explicit before/after raster
        # pair. The planner stores these paths in step.parameters, while
        # the Specialist contract receives raster inputs separately.
        specialist_inputs = list(inputs or [])

        if (
            step.operation == "temporal_analysis"
            and not specialist_inputs
        ):
            before = step.parameters.get("before")
            after = step.parameters.get("after")

            if before and after:
                specialist_inputs = [
                    str(before),
                    str(after),
                ]

        try:
            evidence = specialist.infer(
                specialist_inputs,
                parameters=step.parameters,
            )

            # When orchestration selected a concrete model, make that selection
            # authoritative in the evidence record.
            if selected_model:
                evidence = evidence.model_copy(
                    update={"model": selected_model}
                )

            self.evidence_registry.add(evidence)
        finally:
            # Learned specialists may retain CUDA allocations. Evidence holds
            # CPU-native results, so release the model after registration.
            unload = getattr(specialist, "unload", None)
            if callable(unload):
                unload()

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

        vqa_consistency = None

        # Evidence-conditioned VQA receives its structured evidence
        # through the VQA Evidence contract. Only VQA evidence is
        # evaluated by the structured consistency layer.
        if expected_task == "vqa":
            vqa_evidence = next(
                (
                    item
                    for item in reversed(evidence)
                    if item.task == "vqa"
                ),
                None,
            )

            if vqa_evidence is not None:
                answer = vqa_evidence.result.get("answer")
                structured_evidence = (
                    vqa_evidence.result.get(
                        "vqa_structured_evidence"
                    )
                )

                if (
                    isinstance(answer, str)
                    and isinstance(structured_evidence, dict)
                ):
                    vqa_consistency = evaluate_vqa_consistency(
                        answer=answer,
                        evidence=structured_evidence,
                    )

        verification = self.verifier.verify(
            evidence,
            expected_task=expected_task,
            required_modalities=required_modalities_value,
            vqa_consistency=vqa_consistency,
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
        dependency_evidence_ids: list[str] | None = None,
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

        if step.operation == "change_geospatialization":
            try:
                if not dependency_evidence_ids:
                    raise ValueError(
                        "change_geospatialization requires dependency Evidence."
                    )

                if len(dependency_evidence_ids) != 1:
                    raise ValueError(
                        "change_geospatialization requires exactly one "
                        "temporal Evidence dependency."
                    )

                source_evidence = self.evidence_registry.get(
                    dependency_evidence_ids[0]
                )

                evidence = self.change_geospatializer.execute(
                    source_evidence
                )

                # ChangeGeospatializer returns a newly created Evidence
                # object. Register it before exposing its ID to downstream
                # dependency-aware execution.
                self.evidence_registry.add(evidence)

                return ExecutionResult(
                    success=True,
                    step_id=step.step_id,
                    task=step.task,
                    output=evidence.model_dump(),
                    evidence_ids=[evidence.evidence_id],
                    message=(
                        "Temporal change raster was converted into "
                        "deterministic geospatial Evidence."
                    ),
                )

            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    step_id=step.step_id,
                    task=step.task,
                    message=(
                        "Change geospatialization failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

        if step.operation == "verification":
            return self.execute_verification(step)

        if step.operation == "multimodal_alignment":
            try:
                evidence = self.multimodal_executor.execute(
                    source_evidence_ids=dependency_evidence_ids,
                    parameters=step.parameters,
                )

                self.evidence_registry.add(evidence)

                return ExecutionResult(
                    success=True,
                    step_id=step.step_id,
                    task=step.task,
                    output=evidence.result,
                    evidence_ids=[evidence.evidence_id],
                    message=(
                        "Deterministic multimodal alignment "
                        "completed."
                    ),
                )

            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    step_id=step.step_id,
                    task=step.task,
                    message=(
                        f"Multimodal alignment failed: {exc}"
                    ),
                )

        if step.operation in {
            "buffer",
            "intersection",
            "distance",
            "area",
        }:
            return self.execute_gis(
                step,
                source_evidence_ids=dependency_evidence_ids,
            )

        return ExecutionResult(
            success=False,
            step_id=step.step_id,
            task=step.task,
            message=(
                f"Execution operation '{step.operation}' is not "
                f"implemented yet."
            ),
        )

    def execute_gis(
        self,
        step: PlanStep,
        source_evidence_ids: list[str] | None = None,
    ) -> ExecutionResult:
        """
        Execute a deterministic GIS operation over dependency-produced Evidence.

        When source evidence IDs are supplied, GIS operates only on those
        explicitly resolved dependencies.
        """
        try:
            evidence = self.gis_executor.execute(
                step.operation,
                parameters=step.parameters,
                source_evidence_ids=source_evidence_ids,
            )

            return ExecutionResult(
                success=True,
                step_id=step.step_id,
                task=step.task,
                output=evidence.result,
                evidence_ids=[evidence.evidence_id],
                message="Deterministic GIS execution completed.",
            )

        except Exception as exc:
            return ExecutionResult(
                success=False,
                step_id=step.step_id,
                task=step.task,
                message=f"GIS execution failed: {exc}",
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

        # Map plan step IDs to the Evidence IDs produced by those steps.
        # This makes GIS dependencies explicit and auditable.
        step_evidence_ids: dict[str, list[str]] = {}

        for step in plan.steps:
            dependency_evidence_ids: list[str] = []

            for dependency_step_id in step.depends_on:
                if dependency_step_id not in step_evidence_ids:
                    return results + [
                        ExecutionResult(
                            success=False,
                            step_id=step.step_id,
                            task=step.task,
                            message=(
                                f"Dependency step '{dependency_step_id}' "
                                f"has not produced Evidence."
                            ),
                        )
                    ]

                dependency_evidence_ids.extend(
                    step_evidence_ids[dependency_step_id]
                )

            result = self.execute_step(
                step,
                inputs,
                specialist_override=specialist_bindings.get(step.task),
                selected_model=selected_models.get(step.task),
                dependency_evidence_ids=dependency_evidence_ids,
            )

            results.append(result)

            if result.success:
                step_evidence_ids[step.step_id] = list(
                    result.evidence_ids
                )

            if not result.success:
                break

        return results
