from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """Structured representation of a user's SATQuery task."""

    task_id: str

    query: str

    task_type: str

    required_capabilities: list[str] = Field(default_factory=list)

    required_modalities: list[str] = Field(default_factory=list)

    input_count: int = Field(default=0, ge=0)

    requires_temporal_pair: bool = False

    spatial_operations: list[str] = Field(default_factory=list)

    parameters: dict[str, str | int | float | bool] = Field(
        default_factory=dict
    )

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
