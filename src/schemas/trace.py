from typing import Any

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """Auditable record of one SATQuery execution step."""

    event_id: str

    step: int = Field(ge=0)

    component: str

    action: str

    status: str

    task_id: str | None = None

    model: str | None = None

    parameters: dict[str, Any] = Field(default_factory=dict)

    input_refs: list[str] = Field(default_factory=list)

    output_refs: list[str] = Field(default_factory=list)

    message: str | None = None
