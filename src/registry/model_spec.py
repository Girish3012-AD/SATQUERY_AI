from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    capability: str
    task_types: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)

    status: str

    checkpoint: str | None = None
    version: str | None = None

    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict
    )
