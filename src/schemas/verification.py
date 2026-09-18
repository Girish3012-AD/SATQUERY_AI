from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Result of validating a SATQuery answer against its evidence."""

    status: str

    verified: bool

    confidence: float = Field(ge=0.0, le=1.0)

    reasons: list[str] = Field(default_factory=list)

    evidence_ids: list[str] = Field(default_factory=list)

    conflicts: list[str] = Field(default_factory=list)

    recommended_action: str | None = None

    # Optional claim-level verification for evidence-conditioned VQA.
    # This is intentionally separate from the overall verification status,
    # which still reflects evidence confidence and required constraints.
    vqa_claim_status: str | None = None
