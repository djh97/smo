from __future__ import annotations

from collections import defaultdict

from .schemas import RecordedVisit


class VisitStore:
    """In-memory session-scoped visit memory."""

    def __init__(self) -> None:
        self._visits: dict[str, list[RecordedVisit]] = defaultdict(list)

    def record_visit(
        self,
        patient_id: str,
        visit_type: str,
        case_text: str,
        timestamp: str,
    ) -> RecordedVisit:
        visit = RecordedVisit(
            patient_id=patient_id,
            visit_type=visit_type,
            timestamp=timestamp,
            case_text=case_text,
        )
        self._visits[patient_id].append(visit)
        self._visits[patient_id].sort(key=lambda item: item.timestamp)
        return visit

    def get_latest_visit(
        self,
        patient_id: str,
        exclude_timestamp: str | None = None,
    ) -> RecordedVisit | None:
        matches = [
            visit
            for visit in self._visits.get(patient_id, [])
            if exclude_timestamp is None or visit.timestamp != exclude_timestamp
        ]
        if not matches:
            return None
        matches.sort(key=lambda item: item.timestamp, reverse=True)
        return matches[0]
