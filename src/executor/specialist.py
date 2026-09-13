from abc import ABC, abstractmethod
from typing import Any

from src.schemas import Evidence


class Specialist(ABC):
    """
    Interface for an actual specialist model/component.

    Concrete implementations must perform real inference and return
    normalized Evidence.
    """

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
