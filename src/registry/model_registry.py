from .model_spec import ModelSpec


VALID_STATUSES = {
    "AVAILABLE",
    "PLANNED",
    "UNAVAILABLE",
}


class ModelRegistry:
    """
    Registry of models/components that can provide SATQuery capabilities.

    Registration is metadata-only. The registry does not load models or
    perform inference.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelSpec] = {}

    def register(self, model: ModelSpec) -> None:
        if model.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid model status: {model.status}. "
                f"Expected one of {sorted(VALID_STATUSES)}"
            )

        if model.name in self._models:
            raise ValueError(
                f"Model already exists: {model.name}"
            )

        self._models[model.name] = model

    def get(self, name: str) -> ModelSpec:
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"Model not found: {name}") from exc

    def all(self) -> list[ModelSpec]:
        return list(self._models.values())

    def count(self) -> int:
        return len(self._models)

    def clear(self) -> None:
        self._models.clear()

    def find_by_capability(
        self,
        capability: str,
        available_only: bool = False,
    ) -> list[ModelSpec]:
        models = [
            model
            for model in self._models.values()
            if model.capability == capability
        ]

        if available_only:
            models = [
                model
                for model in models
                if model.status == "AVAILABLE"
            ]

        return models

    def find(
        self,
        capability: str,
        task_type: str | None = None,
        modality: str | None = None,
        available_only: bool = False,
    ) -> list[ModelSpec]:
        results = self.find_by_capability(
            capability,
            available_only=available_only,
        )

        if task_type is not None:
            results = [
                model
                for model in results
                if task_type in model.task_types
            ]

        if modality is not None:
            results = [
                model
                for model in results
                if modality in model.modalities
            ]

        return results
