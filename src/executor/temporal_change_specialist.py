from __future__ import annotations

from pathlib import Path
from typing import Any

from src.executor.change_specialist import ChangeSpecialist
from src.executor.learned_change_specialist import (
    LearnedChangeSpecialist,
)
from src.executor.specialist import Specialist
from src.schemas.evidence import Evidence


class TemporalChangeSpecialist(Specialist):
    """
    Unified temporal specialist.

    Primary:
        Learned ChangeUNet

    Fallback:
        Existing deterministic raster-difference specialist.

    The fallback is retained deliberately so the system can
    abstain or degrade safely instead of pretending that a
    learned model is universally applicable.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        use_learned: bool = True,
        allow_fallback: bool = True,
    ) -> None:

        self.use_learned = use_learned
        self.allow_fallback = allow_fallback

        self.learned = (
            LearnedChangeSpecialist(
                checkpoint_path=checkpoint_path
            )
            if use_learned
            else None
        )

        self.fallback = ChangeSpecialist()

    @property
    def capability(self) -> str:
        return "temporal_analysis"

    def infer(
        self,
        inputs: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> Evidence:

        parameters = parameters or {}

        if len(inputs) != 2:
            raise ValueError(
                "TemporalChangeSpecialist requires "
                "exactly two raster inputs."
            )

        learned_error = None

        if self.use_learned and self.learned is not None:

            try:
                evidence = self.learned.infer(
                    inputs,
                    parameters,
                )

                evidence.metadata[
                    "routing_mode"
                ] = "learned_primary"

                evidence.metadata[
                    "fallback_available"
                ] = self.allow_fallback

                return evidence

            except Exception as exc:
                learned_error = str(exc)

                if not self.allow_fallback:
                    raise RuntimeError(
                        "Learned temporal specialist failed "
                        "and fallback is disabled: "
                        f"{exc}"
                    ) from exc

        if self.allow_fallback:

            evidence = self.fallback.infer(
                inputs,
                parameters,
            )

            evidence.metadata[
                "routing_mode"
            ] = "deterministic_fallback"

            evidence.metadata[
                "learned_error"
            ] = learned_error

            return evidence

        raise RuntimeError(
            "No temporal specialist is available."
        )
