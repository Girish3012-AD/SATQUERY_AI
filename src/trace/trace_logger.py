import json
import time
from pathlib import Path
from typing import Any


class TraceLogger:
    """Records observable SATQuery pipeline events."""

    def __init__(self, output_dir: str = "outputs/traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []

    def log_step(
        self,
        step_number: int,
        component: str,
        action: str,
        details: dict[str, Any],
    ) -> None:
        """Record one observable pipeline event."""
        self.events.append(
            {
                "step": step_number,
                "timestamp": time.time(),
                "component": component,
                "action": action,
                "details": details,
            }
        )

    def export_trace(self, task_id: str) -> Path:
        """Export the recorded events as a machine-readable JSON trace."""
        trace_payload = {
            "trace_id": f"TRACE_{task_id}",
            "generated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "total_steps": len(self.events),
            "execution_trace": self.events,
        }

        file_path = self.output_dir / f"trace_{task_id}.json"

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                trace_payload,
                file,
                indent=2,
            )

        return file_path
