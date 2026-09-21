from typing import Any

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    capability: str
    task_types: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)

    status: str

    specialist_name: str | None = None

    checkpoint: str | None = None
    version: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )
