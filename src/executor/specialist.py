from abc import ABC, abstractmethod
from typing import Any

from src.schemas import Evidence


class Specialist(ABC):
    """
    Interface for an actual specialist model/component.

    Concrete implementations must perform real inference and return
    normalized Evidence.
    """
    
    # Declarative requirement for the Input Binding router.
    # E.g., {"modality": "optical", "count": 2, "temporal": "bi-temporal"}
    REQUIRED_INPUT_PROFILE: dict[str, Any] | None = None

    @property
    @abstractmethod
    def capability(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        inputs: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> Evidence:
        raise NotImplementedError
