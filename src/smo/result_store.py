from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import SavedCaseResult


class SavedResultStore:
    """Disk-backed store for previously generated patient outputs."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, patient_id: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", patient_id.strip())
        return normalized or "unknown-patient"

    def _path_for(self, patient_id: str) -> Path:
        return self.root_dir / f"{self._safe_name(patient_id)}.json"

    def _read_records(self, patient_id: str) -> list[SavedCaseResult]:
        path = self._path_for(patient_id)
        if not path.exists():
            return []

        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [SavedCaseResult.from_dict(item) for item in payload.get("records", [])]
        records.sort(key=lambda item: item.timestamp)
        return records

    def save(self, record: SavedCaseResult) -> SavedCaseResult:
        records = self._read_records(record.patient_id)
        records.append(record)
        records.sort(key=lambda item: item.timestamp)

        path = self._path_for(record.patient_id)
        payload = {
            "patient_id": record.patient_id,
            "records": [item.to_dict() for item in records],
        }
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)
        return record

    def list_records(self, patient_id: str) -> list[SavedCaseResult]:
        return self._read_records(patient_id)

    def get_latest(self, patient_id: str) -> SavedCaseResult | None:
        records = self._read_records(patient_id)
        if not records:
            return None
        return records[-1]

    def get_by_timestamp(self, patient_id: str, timestamp: str) -> SavedCaseResult | None:
        for record in self._read_records(patient_id):
            if record.timestamp == timestamp:
                return record
        return None
