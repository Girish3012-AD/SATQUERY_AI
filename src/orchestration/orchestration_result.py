from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestrationResult:
    """
    Auditable result produced by the SATQuery orchestration layer.

    The orchestrator coordinates existing controller, planner,
    execution, and verification components. It does not perform
    specialist inference itself.
    """

    success: bool
    task_id: str
    query: str
    task_type: str
    plan_id: str
    executed_steps: list[str] = field(default_factory=list)
    successful_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    selected_capabilities: dict[str, str] = field(default_factory=dict)
    selected_models: dict[str, str] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.verification.get("status") == "verified":
            return "verified"

        if self.verification.get("status") == "low_confidence":
            return "low_confidence"

        if self.verification.get("status") == "abstain":
            return "abstain"

        if self.success:
            return "completed"

        return "failed"
