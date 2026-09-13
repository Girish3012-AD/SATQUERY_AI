from typing import Any

from pydantic import BaseModel, Field


class ImageRef(BaseModel):
    """
    Structured description of a remote-sensing image.

    This object is produced by the input validation layer and
    passed to downstream SATQuery components.
    """

    path: str
    filename: str

    format: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    band_count: int = Field(gt=0)

    dtype: str

    crs: str | None = None
    resolution_x: float | None = None
    resolution_y: float | None = None

    bounds: tuple[float, float, float, float] | None = None

    modality: str | None = None
    sensor: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
