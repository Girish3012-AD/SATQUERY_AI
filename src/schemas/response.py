from typing import Any

from pydantic import BaseModel, Field

from .evidence import Evidence
from .verification import VerificationResult


class FinalResponse(BaseModel):
    """User-facing SATQuery result."""

    answer: str

    status: str

    confidence: float = Field(ge=0.0, le=1.0)

    evidence: list[Evidence] = Field(default_factory=list)

    verification: VerificationResult | None = None

    visualizations: list[str] = Field(default_factory=list)

    report_path: str | None = None

    trace_id: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
