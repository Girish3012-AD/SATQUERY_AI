from dataclasses import dataclass, field

from src.schemas import TaskSpec


@dataclass
class PlanStep:
    step_id: str
    task: str
    operation: str
    depends_on: list[str] = field(default_factory=list)
    parameters: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass
class EvidencePlan:
    plan_id: str
    task_id: str
    query: str
    steps: list[PlanStep] = field(default_factory=list)

    def get_step(self, step_id: str) -> PlanStep:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        raise KeyError(f"Plan step not found: {step_id}")

    def step_ids(self) -> list[str]:
        return [step.step_id for step in self.steps]


class EvidencePlanner:
    """
    Converts a TaskSpec into a deterministic executable evidence plan.

    The planner decides which evidence-producing operations are required
    and establishes dependencies between them. It does not perform the
    actual model inference or GIS operations.
    """

    def create_plan(self, task_spec: TaskSpec) -> EvidencePlan:
        steps: list[PlanStep] = []

        capabilities = set(task_spec.required_capabilities)

        # Temporal analysis
        if task_spec.requires_temporal_pair or "temporal_analysis" in capabilities:
            steps.append(
                PlanStep(
                    step_id="T1",
                    task="temporal_analysis",
                    operation="temporal_analysis",
                )
            )

        # Specialist detection tasks
        specialist_capabilities = [
            "flood_detection",
            "building_detection",
            "water_detection",
            "vegetation_detection",
            "crop_detection",
            "road_detection",
        ]

        for capability in specialist_capabilities:
            if capability in capabilities:
                steps.append(
                    PlanStep(
                        step_id=f"T{len(steps) + 1}",
                        task=capability,
                        operation="specialist_inference",
                    )
                )

        # SAR analysis
        if "sar_analysis" in capabilities:
            steps.append(
                PlanStep(
                    step_id=f"T{len(steps) + 1}",
                    task="sar_analysis",
                    operation="sar_analysis",
                    parameters=dict(task_spec.parameters),
                )
            )

        # Generic VQA fallback
        if (
            not steps
            and "vqa" in capabilities
        ):
            steps.append(
                PlanStep(
                    step_id="T1",
                    task="vqa",
                    operation="visual_question_answering",
                    parameters={
                        "query": task_spec.query,
                    },
                )
            )

        # Multimodal optical + SAR composition
        #
        # The planner only creates this step when the TaskSpec
        # explicitly requires both modalities. The specialist
        # steps must execute first so their Evidence IDs can be
        # supplied as dependencies to the multimodal executor.
        required_modalities = set(
            task_spec.required_modalities
        )

        if {
            "optical",
            "sar",
        }.issubset(required_modalities):
            specialist_step_ids = [
                step.step_id
                for step in steps
                if step.operation
                in {
                    "specialist_inference",
                    "sar_analysis",
                    "temporal_analysis",
                }
            ]

            if specialist_step_ids:
                steps.append(
                    PlanStep(
                        step_id=f"T{len(steps) + 1}",
                        task="multimodal_alignment",
                        operation="multimodal_alignment",
                        depends_on=specialist_step_ids,
                        parameters=dict(
                            task_spec.parameters
                        ),
                    )
                )

        # Spatial operations
        previous_ids = [step.step_id for step in steps]

        for operation in task_spec.spatial_operations:
            step_id = f"T{len(steps) + 1}"

            parameters = dict(task_spec.parameters)

            steps.append(
                PlanStep(
                    step_id=step_id,
                    task=operation,
                    operation=operation,
                    depends_on=previous_ids.copy(),
                    parameters=parameters,
                )
            )

            previous_ids = [step_id]

        # Verification is always the final planning stage.
        if steps:
            steps.append(
                PlanStep(
                    step_id=f"T{len(steps) + 1}",
                    task="verification",
                    operation="verification",
                    depends_on=[steps[-1].step_id],
                )
            )

        return EvidencePlan(
            plan_id=f"PLAN-{task_spec.task_id}",
            task_id=task_spec.task_id,
            query=task_spec.query,
            steps=steps,
        )
