from dataclasses import dataclass, field

from src.registry import ModelRegistry, ModelSpec
from src.schemas import TaskSpec


@dataclass
class RouteResult:
    routed: bool
    capability: str
    modality: str | None
    model: ModelSpec | None = None
    reason: str = ""
    alternatives: list[str] = field(default_factory=list)


class SensorAwareRouter:
    """
    Selects an AVAILABLE registered model compatible with a TaskSpec.

    The router does not load models or execute inference.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def route(
        self,
        task_spec: TaskSpec,
        capability: str,
        modality: str | None = None,
    ) -> RouteResult:
        candidates = self.registry.find(
            capability=capability,
            task_type=task_spec.task_type,
            modality=modality,
            available_only=True,
        )

        if not candidates:
            return RouteResult(
                routed=False,
                capability=capability,
                modality=modality,
                reason=(
                    f"No AVAILABLE model supports capability="
                    f"'{capability}', task_type="
                    f"'{task_spec.task_type}', modality="
                    f"'{modality}'."
                ),
            )

        selected = candidates[0]

        alternatives = [
            model.name
            for model in candidates[1:]
        ]

        return RouteResult(
            routed=True,
            capability=capability,
            modality=modality,
            model=selected,
            reason=f"Selected model '{selected.name}'.",
            alternatives=alternatives,
        )
