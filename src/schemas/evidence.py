from typing import Any

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Standardized evidence produced by SATQuery specialists."""

    evidence_id: str

    source: str

    task: str

    model: str

    sensor: str | None = None

    modality: str | None = None

    timestamp: str | None = None

    geometry: dict[str, Any] | None = None

    measurement: dict[str, Any] | None = None

    result: Any = None

    confidence: float = Field(ge=0.0, le=1.0)

    provenance: dict[str, Any] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)
