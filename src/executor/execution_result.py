from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    success: bool
    step_id: str
    task: str
    output: Any = None
    evidence_ids: list[str] = field(default_factory=list)
    message: str = ""
