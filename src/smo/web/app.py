from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import get_settings
from ..pipeline import AgenticSMOService
from ..reference_data import TEST_CASES
from ..schemas import PatientVisitInput


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

settings = get_settings()
service = AgenticSMOService(settings)

app = FastAPI(title="Agentic Second Medical Opinion")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _default_form_values() -> dict[str, object]:
    case = TEST_CASES[0]
    return {
        "visit_type": case["visit_type"],
        "patient_id": "",
        "age": case["age"],
        "weight": case["weight"],
        "symptoms": case["symptoms"],
        "spo2": case["spo2"],
        "heart_rate": case["heart_rate"],
        "history": case["history"],
    }


def _build_context(
    request: Request,
    *,
    form_values: dict[str, object] | None = None,
    result=None,
    error: str | None = None,
    info: str | None = None,
    saved_visits=None,
    saved_patient_id: str = "",
    selected_saved_timestamp: str = "",
) -> dict[str, object]:
    return {
        "request": request,
        "form_values": form_values or _default_form_values(),
        "result": result,
        "error": error,
        "info": info,
        "saved_visits": saved_visits or [],
        "saved_patient_id": saved_patient_id,
        "selected_saved_timestamp": selected_saved_timestamp,
        **_status_context(),
    }


def _status_context() -> dict[str, object]:
    status = service.runtime_status()
    warnings: list[str] = []
    if not status["openai_configured"]:
        warnings.append("OPENAI_API_KEY is missing.")
    if not status["guideline_files"]:
        warnings.append("No guideline PDFs were found under data/guidelines/.")
    if not status["anthropic_configured"]:
        warnings.append("ANTHROPIC_API_KEY is missing, so Claude will be unavailable.")
    if not status["google_configured"]:
        warnings.append("GOOGLE_API_KEY is missing, so Gemini will be unavailable.")
    return {"runtime_status": status, "runtime_warnings": warnings}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    context = _build_context(request)
    return templates.TemplateResponse("index.html", context)


@app.get("/export", response_class=HTMLResponse)
async def export_saved_result(
    request: Request,
    patient_id: str = "",
    timestamp: str = "",
) -> HTMLResponse:
    raw_patient_id = patient_id.strip()
    raw_timestamp = timestamp.strip()
    error = None
    result = None

    if not raw_patient_id:
        error = "Enter a patient ID to open the paper view."
    else:
        saved = (
            service.load_saved_result_by_timestamp(raw_patient_id, raw_timestamp)
            if raw_timestamp
            else service.load_saved_result(raw_patient_id)
        )
        if saved is None:
            if raw_timestamp:
                error = (
                    f"No saved visit matched patient ID '{raw_patient_id}' "
                    f"at timestamp '{raw_timestamp}'."
                )
            else:
                error = f"No saved visits were found for patient ID '{raw_patient_id}'."
        else:
            result = service.saved_case_to_pipeline_result(saved)

    return templates.TemplateResponse(
        "export.html",
        {
            "request": request,
            "result": result,
            "error": error,
        },
    )


@app.post("/submit", response_class=HTMLResponse)
async def submit(
    request: Request,
    visit_type: str = Form(...),
    patient_id: str = Form(""),
    age: float = Form(...),
    weight: float = Form(...),
    symptoms: str = Form(...),
    spo2: float = Form(...),
    heart_rate: float = Form(...),
    history: str = Form(...),
) -> HTMLResponse:
    form_values = {
        "visit_type": visit_type,
        "patient_id": patient_id,
        "age": age,
        "weight": weight,
        "symptoms": symptoms,
        "spo2": spo2,
        "heart_rate": heart_rate,
        "history": history,
    }

    result = None
    error = None
    saved_visits = []
    try:
        visit_input = PatientVisitInput(
            visit_type=visit_type,
            patient_id=patient_id,
            age=age,
            weight=weight,
            symptoms=symptoms,
            spo2=spo2,
            heart_rate=heart_rate,
            history=history,
        )
        result = service.process_visit(visit_input)
        saved_visits = service.list_saved_results(result.patient_id)
    except Exception as exc:  # pragma: no cover - UI path
        error = str(exc)

    context = _build_context(
        request,
        form_values=form_values,
        result=result,
        error=error,
        saved_visits=saved_visits,
        saved_patient_id=result.patient_id if result else patient_id.strip(),
        selected_saved_timestamp=result.timestamp if result else "",
    )
    return templates.TemplateResponse("index.html", context)


@app.post("/browse", response_class=HTMLResponse)
async def browse_saved_results(
    request: Request,
    patient_id: str = Form(""),
    visit_type: str = Form("New"),
    age: str = Form(""),
    weight: str = Form(""),
    symptoms: str = Form(""),
    spo2: str = Form(""),
    heart_rate: str = Form(""),
    history: str = Form(""),
) -> HTMLResponse:
    form_values = {
        "visit_type": visit_type,
        "patient_id": patient_id,
        "age": age,
        "weight": weight,
        "symptoms": symptoms,
        "spo2": spo2,
        "heart_rate": heart_rate,
        "history": history,
    }

    raw_patient_id = patient_id.strip()
    if not raw_patient_id:
        context = _build_context(
            request,
            form_values=form_values,
            error="Enter a patient ID to browse saved visits.",
        )
        return templates.TemplateResponse("index.html", context)

    saved_visits = service.list_saved_results(raw_patient_id)
    if not saved_visits:
        context = _build_context(
            request,
            form_values=form_values,
            error=f"No saved visits were found for patient ID '{raw_patient_id}'.",
        )
        return templates.TemplateResponse("index.html", context)

    context = _build_context(
        request,
        form_values=saved_visits[0].to_form_values(),
        saved_visits=saved_visits,
        saved_patient_id=raw_patient_id,
        info=f"Found {len(saved_visits)} saved visit(s) for patient ID '{raw_patient_id}'. Select one to open it without making an API call.",
    )
    return templates.TemplateResponse("index.html", context)


@app.post("/load", response_class=HTMLResponse)
async def load_saved_result(
    request: Request,
    patient_id: str = Form(""),
    timestamp: str = Form(""),
) -> HTMLResponse:
    raw_patient_id = patient_id.strip()
    raw_timestamp = timestamp.strip()

    saved_visits = service.list_saved_results(raw_patient_id)
    if not raw_patient_id:
        context = _build_context(
            request,
            error="Enter a patient ID to load a saved visit.",
        )
        return templates.TemplateResponse("index.html", context)

    if not raw_timestamp:
        context = _build_context(
            request,
            saved_visits=saved_visits,
            saved_patient_id=raw_patient_id,
            error="Select a saved visit to open it.",
        )
        return templates.TemplateResponse("index.html", context)

    saved = service.load_saved_result_by_timestamp(raw_patient_id, raw_timestamp)
    if saved is None:
        context = _build_context(
            request,
            saved_visits=saved_visits,
            saved_patient_id=raw_patient_id,
            error=f"No saved visit matched patient ID '{raw_patient_id}' at timestamp '{raw_timestamp}'.",
        )
        return templates.TemplateResponse("index.html", context)

    result = service.saved_case_to_pipeline_result(saved)
    context = _build_context(
        request,
        form_values=saved.to_form_values(),
        result=result,
        info=f"Loaded the saved {saved.visit_type.lower()} visit for patient ID '{saved.patient_id}' from {saved.timestamp}. No API call was made.",
        saved_visits=saved_visits,
        saved_patient_id=saved.patient_id,
        selected_saved_timestamp=saved.timestamp,
    )
    return templates.TemplateResponse("index.html", context)
