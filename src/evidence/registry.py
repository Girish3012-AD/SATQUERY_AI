from src.schemas import Evidence


class EvidenceRegistry:
    """In-memory registry for SATQuery evidence."""

    def __init__(self) -> None:
        self._evidence: dict[str, Evidence] = {}

    def add(self, evidence: Evidence) -> None:
        """Add evidence to the registry."""

        if evidence.evidence_id in self._evidence:
            return  # Idempotent add if already registered

        self._evidence[evidence.evidence_id] = evidence

    def register(self, evidence: Evidence) -> None:
        """Alias for add()."""
        self.add(evidence)

    def get(self, evidence_id: str) -> Evidence:
        """Retrieve evidence by ID."""

        try:
            return self._evidence[evidence_id]
        except KeyError as exc:
            raise KeyError(
                f"Evidence not found: {evidence_id}"
            ) from exc

    def get_by_task(self, task: str) -> list[Evidence]:
        """Retrieve all evidence belonging to a task."""

        return [
            evidence
            for evidence in self._evidence.values()
            if evidence.task == task
        ]

    def get_by_modality(self, modality: str) -> list[Evidence]:
        """Retrieve evidence belonging to a modality."""

        return [
            evidence
            for evidence in self._evidence.values()
            if evidence.modality == modality
        ]

    def all(self) -> list[Evidence]:
        """Return all registered evidence."""

        return list(self._evidence.values())

    def count(self) -> int:
        """Return number of registered evidence objects."""

        return len(self._evidence)

    def clear(self) -> None:
        """Remove all evidence."""

        self._evidence.clear()
