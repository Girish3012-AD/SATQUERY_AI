from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.controller import TaskController
from src.data.input_metadata import InputMetadataResolver
from src.evidence import EvidenceRegistry
from src.executor import ExecutionEngine, Specialist
from src.planner import EvidencePlan, EvidencePlanner
from src.registry import ModelRegistry
from src.router import SensorAwareRouter
from src.schemas import TaskSpec


@dataclass
class OrchestratorConfig:
    """
    Configuration for the deterministic agentic orchestration layer.
    """

    stop_on_failure: bool = True


class SATQueryOrchestrator:
    """
    Coordinates SATQuery's reasoning pipeline.

    Responsibility boundary:

        TaskController  -> understands the query
        EvidencePlanner -> constructs the execution plan
        Orchestrator    -> selects and sequences specialists
        ExecutionEngine -> executes specialists
        GeoReason       -> verifies evidence

    The orchestrator deliberately does not perform model inference
    itself and does not replace the existing execution engine.
    """

    def __init__(
        self,
        *,
        controller: TaskController | None = None,
        planner: EvidencePlanner | None = None,
        evidence_registry: EvidenceRegistry | None = None,
        engine: ExecutionEngine | None = None,
        specialists: Iterable[Specialist] | None = None,
        registry: ModelRegistry | None = None,
        router: SensorAwareRouter | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.controller = controller or TaskController()
        self.planner = planner or EvidencePlanner()
        self.input_metadata_resolver = InputMetadataResolver()
        self.evidence_registry = evidence_registry or EvidenceRegistry()
        self.engine = engine or ExecutionEngine(self.evidence_registry)

        # Registry and router are optional for backwards compatibility.
        # When supplied, routing becomes authoritative for model selection.
        self.registry = registry
        self.router = router or (
            SensorAwareRouter(registry) if registry is not None else None
        )

        self.config = config or OrchestratorConfig()

        # Primary capability registry retained for backwards compatibility.
        self._specialists: dict[str, Specialist] = {}

        # Model-aware registry. Multiple specialist implementations may expose
        # the same capability, with ModelSpec.specialist_name selecting one.
        self._specialists_by_name: dict[str, Specialist] = {}

        if specialists is not None:
            for specialist in specialists:
                self.register_specialist(specialist)

    def register_specialist(self, specialist: Specialist) -> None:
        """
        Register a specialist implementation.

        The legacy capability registry keeps the first implementation for
        backwards-compatible capability-based execution. A separate
        name-based registry allows multiple implementations of the same
        capability to coexist for model-aware routing.
        """
        capability = specialist.capability
        specialist_name = specialist.__class__.__name__

        if specialist_name in self._specialists_by_name:
            existing = self._specialists_by_name[specialist_name]

            # The same implementation class may legitimately serve
            # different capabilities in tests or lightweight adapters.
            # Treat class name + capability as the unique binding identity.
            if existing.capability == capability:
                raise ValueError(
                    f"Specialist already registered: "
                    f"{specialist_name} ({capability})"
                )

            specialist_name = f"{specialist_name}:{capability}"

        self._specialists_by_name[specialist_name] = specialist

        # Preserve the first specialist as the legacy capability fallback.
        if capability not in self._specialists:
            self._specialists[capability] = specialist

            # Keep the existing ExecutionEngine capability registry
            # authoritative for backwards-compatible execution.
            self.engine.register_specialist(specialist)

    @property
    def specialists(self) -> dict[str, Specialist]:
        """
        Return the legacy capability -> specialist mapping.

        Use model-aware routing to select among multiple implementations
        sharing the same capability.
        """
        return dict(self._specialists)

    def build_task_spec(
        self,
        query: str,
        input_count: int,
        parameters: dict | None = None,
    ) -> TaskSpec:
        task_spec = self.controller.build_task_spec(
            query=query,
            input_count=input_count,
        )

        if parameters:
            task_spec.parameters.update(parameters)

        return task_spec

    def build_plan(self, task_spec: TaskSpec) -> EvidencePlan:
        return self.planner.create_plan(task_spec)

    def _required_specialist_capabilities(
        self,
        task_spec: TaskSpec,
        plan: EvidencePlan,
    ) -> list[str]:
        """
        Determine the specialist capabilities actually required by the plan.

        Verification and GIS operations are orchestration/execution steps,
        not specialist capabilities.
        """
        specialist_tasks = {
            "vqa",
            "building_detection",
            "flood_detection",
            "water_detection",
            "vegetation_detection",
            "crop_detection",
            "road_detection",
            "temporal_analysis",
            "sar_analysis",
        }

        capabilities: list[str] = []

        for step_id in plan.step_ids():
            step = plan.get_step(step_id)

            if step.task in specialist_tasks and step.task not in capabilities:
                capabilities.append(step.task)

        # Preserve capabilities explicitly declared by the controller.
        for capability in task_spec.required_capabilities:
            if (
                capability in self._specialists
                and capability not in capabilities
            ):
                capabilities.append(capability)

        return capabilities

    def _check_specialist_availability(
        self,
        task_spec: TaskSpec,
        plan: EvidencePlan,
    ) -> tuple[bool, dict[str, str], list[str]]:
        """
        Check whether every specialist required by the generated plan
        has a registered implementation.
        """
        required = self._required_specialist_capabilities(
            task_spec,
            plan,
        )

        selected: dict[str, str] = {}
        missing: list[str] = []

        for capability in required:
            specialist = self._specialists.get(capability)

            if specialist is None:
                missing.append(capability)
            else:
                selected[capability] = specialist.__class__.__name__

        return len(missing) == 0, selected, missing

    def _route_specialists(
        self,
        task_spec: TaskSpec,
        capabilities: list[str],
    ) -> tuple[
        bool,
        dict[str, str],
        dict[str, str],
        dict[str, Specialist],
        list[str],
    ]:
        """
        Route every required capability through the ModelRegistry and
        SensorAwareRouter when routing is configured.

        Returns:
            available,
            selected_specialists,
            selected_models,
            routing_errors
        """
        selected_specialists: dict[str, str] = {}
        selected_models: dict[str, str] = {}
        selected_bindings: dict[str, Specialist] = {}
        routing_errors: list[str] = []

        # Backwards-compatible path for tests or callers that have not
        # configured a registry yet.
        if self.router is None:
            for capability in capabilities:
                specialist = self._specialists.get(capability)

                if specialist is None:
                    routing_errors.append(
                        f"No specialist implementation is registered "
                        f"for capability '{capability}'."
                    )
                    continue

                selected_specialists[capability] = (
                    specialist.__class__.__name__
                )

            return (
                len(routing_errors) == 0,
                selected_specialists,
                selected_models,
                selected_bindings,
                routing_errors,
            )

        for capability in capabilities:
            specialist = self._specialists.get(capability)

            if specialist is None:
                routing_errors.append(
                    f"No specialist implementation is registered "
                    f"for capability '{capability}'."
                )
                continue

            modality = None

            # Use the explicitly requested modality when exactly one
            # modality is required. The router can then enforce it.
            if len(task_spec.required_modalities) == 1:
                modality = task_spec.required_modalities[0]

            route = self.router.route(
                task_spec=task_spec,
                capability=capability,
                modality=modality,
            )

            if not route.routed or route.model is None:
                routing_errors.append(
                    f"Routing failed for capability '{capability}': "
                    f"{route.reason}"
                )
                continue

            selected_model_name = route.model.name

            bound_specialist = self._resolve_model_specialist(
                capability=capability,
                model_name=selected_model_name,
            )

            if bound_specialist is None:
                routing_errors.append(
                    f"Model '{selected_model_name}' is routed for "
                    f"capability '{capability}', but its specialist "
                    f"binding is unavailable."
                )
                continue

            selected_specialists[capability] = (
                bound_specialist.__class__.__name__
            )
            selected_models[capability] = selected_model_name
            selected_bindings[capability] = bound_specialist

        return (
            len(routing_errors) == 0,
            selected_specialists,
            selected_models,
            selected_bindings,
            routing_errors,
        )

    def _resolve_model_specialist(
        self,
        capability: str,
        model_name: str,
    ) -> Specialist | None:
        """
        Resolve the concrete specialist implementation for a routed model.

        ModelSpec.specialist_name is the authoritative binding when supplied.
        Older registry entries without specialist_name retain the existing
        capability-based fallback.
        """
        if self.registry is None:
            return self._specialists.get(capability)

        model = self.registry.get(model_name)

        if model is None:
            return None

        if model.specialist_name:
            specialist = self._specialists_by_name.get(
                model.specialist_name
            )

            if specialist is None:
                specialist = self._specialists_by_name.get(
                    f"{model.specialist_name}:{capability}"
                )

            if specialist is None:
                return None

            if specialist.capability != capability:
                return None

            return specialist

        # Backwards-compatible registry entry.
        return self._specialists.get(capability)

    @staticmethod
    def _collect_evidence_ids(results) -> list[str]:
        evidence_ids: list[str] = []

        for result in results:
            for evidence_id in result.evidence_ids:
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)

        return evidence_ids

    def run(
        self,
        query: str,
        inputs: list[str] | None = None,
    ):
        """
        Execute the complete SATQuery orchestration pipeline.

        Returns OrchestrationResult.
        """
        from src.orchestration.orchestration_result import (
            OrchestrationResult,
        )

        inputs = list(inputs or [])

        input_parameters = (
            self.input_metadata_resolver.resolve(inputs)
        )

        task_spec = self.build_task_spec(
            query=query,
            input_count=len(inputs),
            parameters=input_parameters,
        )

        plan = self.build_plan(task_spec)

        required_capabilities = self._required_specialist_capabilities(
            task_spec,
            plan,
        )

        (
            available,
            selected,
            selected_models,
            selected_bindings,
            routing_errors,
        ) = self._route_specialists(
            task_spec,
            required_capabilities,
        )

        if not available:
            return OrchestrationResult(
                success=False,
                task_id=task_spec.task_id,
                query=task_spec.query,
                task_type=task_spec.task_type,
                plan_id=plan.plan_id,
                selected_capabilities=selected,
                selected_models=selected_models,
                messages=[
                    "Required specialist implementation(s) unavailable.",
                    *routing_errors,
                ],
            )

        results = self.engine.execute(
            plan,
            inputs=inputs,
            specialist_bindings=selected_bindings,
            selected_models=selected_models,
        )

        executed_steps = [result.step_id for result in results]
        successful_steps = [
            result.step_id
            for result in results
            if result.success
        ]
        failed_steps = [
            result.step_id
            for result in results
            if not result.success
        ]

        evidence_ids = self._collect_evidence_ids(results)

        verification = {}

        for result in results:
            if result.task == "verification":
                verification = dict(result.output or {})

        success = (
            len(failed_steps) == 0
            and bool(results)
        )

        messages = [
            result.message
            for result in results
            if result.message
        ]

        return OrchestrationResult(
            success=success,
            task_id=task_spec.task_id,
            query=task_spec.query,
            task_type=task_spec.task_type,
            plan_id=plan.plan_id,
            executed_steps=executed_steps,
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            evidence_ids=evidence_ids,
            verification=verification,
            selected_capabilities=selected,
            selected_models=selected_models,
            messages=messages,
        )

    execute = run
