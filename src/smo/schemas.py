from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class PatientVisitInput:
    visit_type: str
    patient_id: str | None
    age: float
    weight: float
    symptoms: str
    spo2: float
    heart_rate: float
    history: str

    def normalized_visit_type(self) -> str:
        return self.visit_type.strip().lower()

    def resolved_patient_id(self) -> str:
        raw = (self.patient_id or "").strip()
        return raw if raw else f"mso-{uuid4().hex[:8]}"

    def to_case_text(self, patient_id: str) -> str:
        visit_type = self.visit_type.strip().title()
        return (
            f"Patient ID: {patient_id}\n"
            f"Visit Type: {visit_type}\n"
            f"Age: {self.age}, Weight: {self.weight} kg\n"
            f"Symptoms: {self.symptoms.strip()}\n"
            f"SpO2: {self.spo2}%, Heart Rate: {self.heart_rate} bpm\n"
            f"Medical History: {self.history.strip()}"
        )


@dataclass
class RecordedVisit:
    patient_id: str
    visit_type: str
    timestamp: str
    case_text: str


@dataclass
class PromptLogEntry:
    run_id: str
    model: str
    role: str
    tokens_in: int
    tokens_out: int
    duration_seconds: float | None = None
    note: str = ""
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "model": self.model,
            "role": self.role,
            "tokens_in": int(self.tokens_in),
            "tokens_out": int(self.tokens_out),
            "note": self.note,
        }
        if self.duration_seconds is not None:
            payload["duration_seconds"] = float(self.duration_seconds)
        if self.details is not None:
            payload["details"] = self.details
        return payload


@dataclass
class PipelineResult:
    patient_id: str
    timestamp: str
    case_text: str
    visit_summary_html: str
    final_opinion: str
    assessment_html: str
    treatment_plan_html: str
    selected_tools: list[str]
    result_source: str = "live"


@dataclass
class SavedCaseResult:
    patient_id: str
    timestamp: str
    visit_type: str
    age: float
    weight: float
    symptoms: str
    spo2: float
    heart_rate: float
    history: str
    case_text: str
    final_opinion: str
    selected_tools: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "visit_type": self.visit_type,
            "age": self.age,
            "weight": self.weight,
            "symptoms": self.symptoms,
            "spo2": self.spo2,
            "heart_rate": self.heart_rate,
            "history": self.history,
            "case_text": self.case_text,
            "final_opinion": self.final_opinion,
            "selected_tools": list(self.selected_tools),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SavedCaseResult":
        return cls(
            patient_id=str(data["patient_id"]),
            timestamp=str(data["timestamp"]),
            visit_type=str(data["visit_type"]),
            age=float(data["age"]),
            weight=float(data["weight"]),
            symptoms=str(data["symptoms"]),
            spo2=float(data["spo2"]),
            heart_rate=float(data["heart_rate"]),
            history=str(data["history"]),
            case_text=str(data["case_text"]),
            final_opinion=str(data["final_opinion"]),
            selected_tools=[str(item) for item in data.get("selected_tools", [])],
        )

    def to_form_values(self) -> dict[str, object]:
        return {
            "visit_type": self.visit_type,
            "patient_id": self.patient_id,
            "age": self.age,
            "weight": self.weight,
            "symptoms": self.symptoms,
            "spo2": self.spo2,
            "heart_rate": self.heart_rate,
            "history": self.history,
        }
