from __future__ import annotations

import re
from collections import defaultdict
from time import perf_counter
from typing import TYPE_CHECKING, Callable

import tiktoken

from .config import Settings
from .formatting import build_opinion_panels_html, build_visit_summary_html
from .patient_store import VisitStore
from .result_store import SavedResultStore
from .schemas import (
    PatientVisitInput,
    PipelineResult,
    PromptLogEntry,
    SavedCaseResult,
    utcnow_iso,
)
from .text_safety import sanitize_provider_text
from .vectorstore import GuidelineRetriever

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI


ToolCallable = Callable[[str, str], str]


class PromptLogger:
    def __init__(self) -> None:
        self.enabled = True
        self._active_bucket = "default"
        self._buckets: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def activate_bucket(self, name: str) -> None:
        self._active_bucket = name

    def clear_bucket(self, name: str) -> None:
        self._buckets[name].clear()

    def bucket(self, name: str) -> dict[str, list[dict[str, object]]]:
        return self._buckets[name]

    def log(self, entry: PromptLogEntry) -> None:
        if not self.enabled:
            return
        self._buckets[self._active_bucket][entry.run_id].append(entry.to_dict())


class AgenticSMOService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.retriever = GuidelineRetriever(settings)
        self.visit_store = VisitStore()
        self.saved_result_store = SavedResultStore(settings.saved_result_dir)
        self.prompt_logger = PromptLogger()
        self._fallback_encoding = None
        self._llm_openai = None
        self._llm_claude = None
        self._llm_gemini = None

    def runtime_status(self) -> dict[str, object]:
        tool_labels = [spec["label"] for spec in self.available_tool_specs()]
        return {
            "openai_configured": bool(self.settings.openai_api_key),
            "anthropic_configured": bool(self.settings.anthropic_api_key),
            "google_configured": bool(self.settings.google_api_key),
            "guideline_files": [path.name for path in self.retriever.available_guidelines()],
            "tool_labels": tool_labels,
        }

    def ensure_runtime_ready(self) -> None:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings and controller routing.")
        if not self.retriever.available_guidelines():
            raise RuntimeError(
                "No guideline PDFs found. Add the WHO and MSF respiratory PDFs to data/guidelines/ "
                "or set SMO_GUIDELINE_PATHS in .env."
            )

    def available_tool_specs(self) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        if self.settings.openai_api_key:
            specs.append(
                {
                    "name": "gpt4_respiratory_analyst",
                    "label": "GPT-4o",
                    "detail": "GPT-4o RAG",
                    "callable": self.openai_rag_tool,
                }
            )
        if self.settings.anthropic_api_key:
            specs.append(
                {
                    "name": "claude3_respiratory_analyst",
                    "label": "Claude Opus 4.6",
                    "detail": "Claude Opus 4.6 RAG",
                    "callable": self.claude_rag_tool,
                }
            )
        if self.settings.google_api_key:
            specs.append(
                {
                    "name": "gemini_respiratory_analyst",
                    "label": "Gemini 2.5 Flash",
                    "detail": "Gemini 2.5 Flash RAG",
                    "callable": self.gemini_rag_tool,
                }
            )
        return specs

    def _invoke(self, llm: object, prompt: str) -> str:
        safe_prompt = sanitize_provider_text(prompt)
        try:
            response = llm.invoke(safe_prompt)
        except Exception as exc:
            raise RuntimeError(f"Provider request failed: {exc}") from exc
        content = getattr(response, "content", response)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return sanitize_provider_text("\n".join(parts))
        return sanitize_provider_text(str(content))

    def _tokcount_openai(self, text: str, model: str | None = None) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model or self.settings.openai_model)
        except Exception:
            encoding = self._get_fallback_encoding()
            if encoding is None:
                return max(1, len(text) // 4)
        return len(encoding.encode(text))

    def _tokcount_estimate(self, text: str) -> int:
        encoding = self._get_fallback_encoding()
        if encoding is None:
            return max(1, len(text) // 4)
        return len(encoding.encode(text))

    def _get_fallback_encoding(self):
        if self._fallback_encoding is not None:
            return self._fallback_encoding
        try:
            self._fallback_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._fallback_encoding = None
        return self._fallback_encoding

    def _get_openai_llm(self):
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        if self._llm_openai is None:
            from langchain_openai import ChatOpenAI

            self._llm_openai = ChatOpenAI(
                model=self.settings.openai_model,
                temperature=self.settings.temperature,
                api_key=self.settings.openai_api_key,
            )
        return self._llm_openai

    def _get_claude_llm(self):
        if not self.settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
        if self._llm_claude is None:
            from langchain_anthropic import ChatAnthropic

            self._llm_claude = ChatAnthropic(
                model=self.settings.anthropic_model,
                temperature=self.settings.temperature,
                api_key=self.settings.anthropic_api_key,
            )
        return self._llm_claude

    def _get_gemini_llm(self):
        if not self.settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not configured.")
        if self._llm_gemini is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm_gemini = ChatGoogleGenerativeAI(
                model=self.settings.google_model,
                temperature=self.settings.temperature,
                google_api_key=self.settings.google_api_key,
            )
        return self._llm_gemini

    def _log_call(
        self,
        *,
        run_id: str,
        model: str,
        role: str,
        prompt: str,
        output: str,
        tokens_in: int,
        tokens_out: int,
        duration_seconds: float | None = None,
        note: str,
        details: dict[str, object] | None = None,
    ) -> None:
        self.prompt_logger.log(
            PromptLogEntry(
                run_id=run_id,
                model=model,
                role=role,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_seconds=duration_seconds,
                note=note,
                details=details,
            )
        )

    def _log_runtime_event(
        self,
        *,
        run_id: str,
        model: str,
        role: str,
        duration_seconds: float,
        note: str,
        details: dict[str, object] | None = None,
    ) -> None:
        self.prompt_logger.log(
            PromptLogEntry(
                run_id=run_id,
                model=model,
                role=role,
                tokens_in=0,
                tokens_out=0,
                duration_seconds=duration_seconds,
                note=note,
                details=details,
            )
        )

    def _retrieve_context_timed(
        self,
        query_text: str,
        *,
        run_id: str,
        note: str,
    ) -> str:
        started_at = perf_counter()
        rag = self.retriever.retrieve_context(query_text)
        self._log_runtime_event(
            run_id=run_id,
            model="retriever",
            role="retrieval",
            duration_seconds=perf_counter() - started_at,
            note=note,
            details={
                "query_text": query_text,
                "retrieved_context": rag,
            },
        )
        return rag

    def _extract_patient_metadata(self, text: str) -> tuple[str | None, str | None]:
        patient_id_match = re.search(r"Patient ID:\s*(\S+)", text)
        visit_type_match = re.search(r"Visit Type:\s*([^\n]+)", text)
        patient_id = patient_id_match.group(1).strip() if patient_id_match else None
        visit_type = visit_type_match.group(1).strip().lower() if visit_type_match else None
        return patient_id, visit_type

    def _enrich_followup_case(self, input_text: str, style: str) -> str:
        patient_id, visit_type = self._extract_patient_metadata(input_text)
        if visit_type != "follow-up" or not patient_id:
            return input_text

        previous_visit = self.visit_store.get_latest_visit(patient_id)
        if previous_visit is None:
            saved_result = self.saved_result_store.get_latest(patient_id)
            if saved_result is not None:
                previous_visit = self.visit_store.record_visit(
                    patient_id=saved_result.patient_id,
                    visit_type=saved_result.visit_type.lower(),
                    case_text=saved_result.case_text,
                    timestamp=saved_result.timestamp,
                )
        if previous_visit is None:
            return input_text

        if style == "openai":
            preamble = (
                "This is a follow-up visit. Compare the current condition with the previous "
                "one. Evaluate if symptoms have improved, worsened, or remained unchanged. "
                "Adjust the treatment plan accordingly."
            )
        elif style == "claude":
            preamble = (
                "This is a follow-up visit. Compare the patient's condition today with the "
                "previous visit. Indicate any changes in symptoms, diagnosis, or severity. "
                "If treatment has failed, recommend next steps."
            )
        else:
            preamble = (
                "This is a follow-up visit. The patient has received previous treatment. "
                "Please compare this visit to the last one and determine if the symptoms are "
                "improving. Adjust your diagnosis and treatment plan accordingly."
            )

        return (
            f"{preamble}\n\n"
            "== Previous Visit ==\n"
            f"{previous_visit.case_text}\n\n"
            "== Current Visit ==\n"
            f"{input_text}\n"
        )

    def openai_rag_tool(self, input_text: str, run_id: str = "unknown") -> str:
        enriched_input = sanitize_provider_text(self._enrich_followup_case(input_text, style="openai"))
        try:
            rag = self._retrieve_context_timed(
                enriched_input,
                run_id=run_id,
                note="GPT-4o retrieval step",
            )
        except Exception as exc:
            raise RuntimeError(f"GPT-4o retrieval step failed: {exc}") from exc
        prompt = f"""
You are a medical expert following WHO and MSF respiratory guidelines.

{enriched_input}

== Retrieved Guidelines ==
{rag}

Return a structured response with:
1. Is the condition improving, worsening, or unchanged if it is a follow up? If this is a new case with no previous data, say: "Not applicable - first recorded visit."
2. Likely Diagnosis
3. Severity Classification
4. Updated Treatment Plan
""".strip()
        started_at = perf_counter()
        try:
            response = self._invoke(self._get_openai_llm(), prompt)
        except Exception as exc:
            raise RuntimeError(f"GPT-4o tool call failed: {exc}") from exc
        self._log_call(
            run_id=run_id,
            model="gpt-4o",
            role="tool",
            prompt=prompt,
            output=response,
            tokens_in=self._tokcount_openai(prompt, self.settings.openai_model),
            tokens_out=self._tokcount_openai(response, self.settings.openai_model),
            duration_seconds=perf_counter() - started_at,
            note="RAG tool call",
            details={
                "prompt": prompt,
                "output": response,
            },
        )
        return response

    def claude_rag_tool(self, input_text: str, run_id: str = "unknown") -> str:
        enriched_input = sanitize_provider_text(self._enrich_followup_case(input_text, style="claude"))
        try:
            rag = self._retrieve_context_timed(
                enriched_input,
                run_id=run_id,
                note="Claude retrieval step",
            )
        except Exception as exc:
            raise RuntimeError(f"Claude retrieval step failed: {exc}") from exc
        prompt = f"""
You are a medical expert following WHO and MSF respiratory guidelines.

{enriched_input}

== Retrieved Guidelines ==
{rag}

Return a structured medical opinion:
1. Is the condition improving, worsening, or unchanged if it is a follow up? If this is a new case with no previous data, say: "Not applicable - first recorded visit."
2. Diagnosis
3. Severity
4. Treatment Plan Adjustments
""".strip()
        started_at = perf_counter()
        try:
            response = self._invoke(self._get_claude_llm(), prompt)
        except Exception as exc:
            raise RuntimeError(f"Claude tool call failed: {exc}") from exc
        self._log_call(
            run_id=run_id,
            model="claude-opus-4-6",
            role="tool",
            prompt=prompt,
            output=response,
            tokens_in=self._tokcount_estimate(prompt),
            tokens_out=self._tokcount_estimate(response),
            duration_seconds=perf_counter() - started_at,
            note="RAG tool call (estimated tokens)",
            details={
                "prompt": prompt,
                "output": response,
            },
        )
        return response

    def gemini_rag_tool(self, input_text: str, run_id: str = "unknown") -> str:
        enriched_input = sanitize_provider_text(self._enrich_followup_case(input_text, style="gemini"))
        try:
            rag = self._retrieve_context_timed(
                enriched_input,
                run_id=run_id,
                note="Gemini retrieval step",
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini retrieval step failed: {exc}") from exc
        prompt = f"""
You are a medical expert following WHO and MSF respiratory guidelines.

{enriched_input}

== Retrieved Guidelines ==
{rag}

Output a structured response:
1. Is the condition improving, worsening, or unchanged if it is a follow up? If this is a new case with no previous data, say: "Not applicable - first recorded visit."
2. Likely Diagnosis
3. Severity Classification
4. Recommended Treatment Plan
""".strip()
        started_at = perf_counter()
        try:
            response = self._invoke(self._get_gemini_llm(), prompt)
        except Exception as exc:
            raise RuntimeError(f"Gemini tool call failed: {exc}") from exc
        self._log_call(
            run_id=run_id,
            model="gemini-2.5-flash",
            role="tool",
            prompt=prompt,
            output=response,
            tokens_in=self._tokcount_estimate(prompt),
            tokens_out=self._tokcount_estimate(response),
            duration_seconds=perf_counter() - started_at,
            note="RAG tool call (estimated tokens)",
            details={
                "prompt": prompt,
                "output": response,
            },
        )
        return response

    def openai_baseline_tool(self, input_text: str) -> str:
        prompt = f"""
You are a medical expert in respiratory diseases.

{sanitize_provider_text(input_text)}

Return a structured response with:
1. Is the condition improving, worsening, or unchanged if it is a follow up? If this is a new case with no previous data, say: "Not applicable - first recorded visit."
2. Likely Diagnosis
3. Severity Classification
4. Updated Treatment Plan
""".strip()
        return self._invoke(self._get_openai_llm(), prompt)

    def claude_baseline_tool(self, input_text: str) -> str:
        prompt = f"""
You are a medical expert in respiratory diseases.

{sanitize_provider_text(input_text)}

Return a structured medical opinion:
1. Is the condition improving, worsening, or unchanged if it is a follow up? If this is a new case with no previous data, say: "Not applicable - first recorded visit."
2. Diagnosis
3. Severity
4. Treatment Plan Adjustments
""".strip()
        return self._invoke(self._get_claude_llm(), prompt)

    def gemini_baseline_tool(self, input_text: str) -> str:
        prompt = f"""
You are a medical expert in respiratory diseases.

{sanitize_provider_text(input_text)}

Output a structured response:
1. Is the condition improving, worsening, or unchanged if it is a follow up? If this is a new case with no previous data, say: "Not applicable - first recorded visit."
2. Likely Diagnosis
3. Severity Classification
4. Recommended Treatment Plan
""".strip()
        return self._invoke(self._get_gemini_llm(), prompt)

    def run_agentic_combined_rag_synthesis(
        self,
        visit_type: str,
        patient_id: str,
        age: float,
        weight: float,
        symptoms: str,
        spo2: float,
        heart_rate: float,
        history: str,
        *,
        run_id: str | None = None,
    ) -> str:
        self.ensure_runtime_ready()
        case_text = sanitize_provider_text(
            (
            f"Patient ID: {patient_id}\n"
            f"Visit Type: {visit_type}\n"
            f"Age: {age}, Weight: {weight} kg\n"
            f"Symptoms: {symptoms}\n"
            f"SpO2: {spo2}%, Heart Rate: {heart_rate} bpm\n"
            f"Medical History: {history}"
            )
        )
        final_output, _ = self._run_agentic(case_text, run_id=run_id or patient_id)
        return final_output

    def _run_agentic(self, case_text: str, *, run_id: str) -> tuple[str, list[str]]:
        pipeline_started_at = perf_counter()
        try:
            controller_guideline_context = self._retrieve_context_timed(
                case_text,
                run_id=run_id,
                note="Controller retrieval step",
            )
        except Exception as exc:
            raise RuntimeError(f"Controller retrieval step failed: {exc}") from exc

        tool_specs = self.available_tool_specs()
        tool_lines = "\n".join(
            f"- {spec['name']} ({spec['detail']})" for spec in tool_specs
        )
        router_prompt = f"""
You are a routing controller. Decide which expert tools to call for this patient case.

Available tools:
{tool_lines}

Return ONLY a comma-separated list of tool names to call.
Rules:
- Call at least ONE tool.
- Prefer 2-3 tools if the case is severe, ambiguous, or high risk.
- If it is straightforward, 1 tool is acceptable.

== Patient Case ==
{case_text}
""".strip()
        router_started_at = perf_counter()
        try:
            router_output = self._invoke(self._get_openai_llm(), router_prompt)
        except Exception as exc:
            raise RuntimeError(f"OpenAI router step failed: {exc}") from exc
        self._log_call(
            run_id=run_id,
            model="gpt-4o",
            role="controller",
            prompt=router_prompt,
            output=router_output,
            tokens_in=self._tokcount_openai(router_prompt, self.settings.openai_model),
            tokens_out=self._tokcount_openai(router_output, self.settings.openai_model),
            duration_seconds=perf_counter() - router_started_at,
            note="Router decision (dynamic tool selection)",
            details={
                "prompt": router_prompt,
                "output": router_output,
            },
        )

        raw = router_output.lower()
        selected_specs = [spec for spec in tool_specs if spec["name"] in raw]
        if not selected_specs:
            selected_specs = [tool_specs[0]]

        outputs: dict[str, str] = {}
        display_labels: list[str] = []
        for spec in selected_specs:
            tool_fn = spec["callable"]
            try:
                outputs[str(spec["name"])] = sanitize_provider_text(tool_fn(case_text, run_id))
            except Exception as exc:
                raise RuntimeError(f"{spec['label']} failed: {exc}") from exc
            display_labels.append(str(spec["label"]))

        synthesis_prompt = f"""
You are the final medical controller.
You have up to three expert opinions. Some may be missing if they were not called.
Synthesize ONE conservative, guideline-grounded answer.

Controller rules:
- Anchor the final answer to the CURRENT patient case, not to a generic disease template.
- Use the retrieved guideline context as the primary source of truth.
- When expert tools disagree, prefer the recommendation that is most complete AND most specifically supported by the retrieved guidelines and the patient details.
- Do NOT average incompatible recommendations into a generic compromise.
- Preserve patient-specific discriminators, especially age group, SpO2, symptom pattern, and major history/comorbidity details.
- If the case is pediatric, keep the answer explicitly pediatric; if the case is adult/elderly, keep the answer explicitly adult/elderly.
- If severity is high, do not under-treat by omitting clearly indicated acute interventions from the stronger guideline-supported tool output.
- Do NOT add unsupported extrapolations, optional long-term management, or extra investigations unless they are clearly justified by the current case and supported by the retrieved guidelines.
- Prefer specificity over breadth. A shorter but more case-specific and guideline-faithful recommendation is better than a broad generic one.
- When one tool provides a more complete guideline-faithful plan, preserve those concrete details rather than compressing them away.
- Preserve high-value details when they are supported by the retrieved guidelines, especially:
  - explicit dosing or administration details
  - oxygen targets
  - monitoring frequency or key monitoring items
  - escalation criteria
  - hospitalization or admission rationale
  - patient-specific risk factors and severity indicators

Return your final opinion in this exact 4-point format:
1. Condition trend (if follow-up) OR "Not applicable - first recorded visit."
2. Likely Diagnosis
3. Severity Classification
4. Recommended Treatment Plan

In the final answer:
- State the diagnosis and severity in a case-specific way.
- Make the treatment plan patient-specific rather than generic.
- Include the key acute interventions that are directly indicated.
- Avoid listing fallback or escalation options unless they are clearly warranted by the current presentation.
- Under each numbered item, you may use short sub-bullets where helpful.
- For severe or high-risk cases, produce a richer clinical note rather than a compressed summary.
- In the treatment plan, include concise sections when supported by the case:
  - Immediate Management
  - Monitoring
  - Hospitalization / Escalation
  - Post-Stabilization or Follow-up
- Keep the answer readable, but do not sacrifice clinically important detail for brevity.

== Patient Case ==
{case_text}

== Retrieved Guidelines ==
{controller_guideline_context}

== GPT-4o Tool Output ==
{outputs.get("gpt4_respiratory_analyst", "NOT CALLED")}

== Claude Tool Output ==
{outputs.get("claude3_respiratory_analyst", "NOT CALLED")}

== Gemini Tool Output ==
{outputs.get("gemini_respiratory_analyst", "NOT CALLED")}
""".strip()
        synthesis_started_at = perf_counter()
        try:
            final_output = self._invoke(self._get_openai_llm(), synthesis_prompt)
        except Exception as exc:
            raise RuntimeError(f"OpenAI final synthesis failed: {exc}") from exc
        self._log_call(
            run_id=run_id,
            model="gpt-4o",
            role="controller",
            prompt=synthesis_prompt,
            output=final_output,
            tokens_in=self._tokcount_openai(synthesis_prompt, self.settings.openai_model),
            tokens_out=self._tokcount_openai(final_output, self.settings.openai_model),
            duration_seconds=perf_counter() - synthesis_started_at,
            note="Final synthesis (controller)",
            details={
                "prompt": synthesis_prompt,
                "output": final_output,
                "selected_tools": display_labels,
            },
        )
        self._log_runtime_event(
            run_id=run_id,
            model="pipeline",
            role="total",
            duration_seconds=perf_counter() - pipeline_started_at,
            note="End-to-end agentic pipeline runtime",
            details={
                "selected_tools": display_labels,
            },
        )
        return final_output, display_labels

    def _build_pipeline_result(
        self,
        *,
        patient_id: str,
        timestamp: str,
        case_text: str,
        age: float,
        weight: float,
        symptoms: str,
        spo2: float,
        heart_rate: float,
        history: str,
        final_output: str,
        selected_tools: list[str],
        result_source: str,
    ) -> PipelineResult:
        summary_html = build_visit_summary_html(
            patient_id=patient_id,
            timestamp=timestamp,
            age=age,
            weight=weight,
            symptoms=symptoms,
            spo2=spo2,
            heart_rate=heart_rate,
            history=history,
        )
        assessment_html, treatment_plan_html = build_opinion_panels_html(
            final_output,
            selected_tools=selected_tools,
        )
        return PipelineResult(
            patient_id=patient_id,
            timestamp=timestamp,
            case_text=case_text,
            visit_summary_html=summary_html,
            final_opinion=final_output,
            assessment_html=assessment_html,
            treatment_plan_html=treatment_plan_html,
            selected_tools=selected_tools,
            result_source=result_source,
        )

    def load_saved_result(self, patient_id: str) -> SavedCaseResult | None:
        raw = patient_id.strip()
        if not raw:
            return None
        saved = self.saved_result_store.get_latest(raw)
        if saved is None:
            return None
        self.visit_store.record_visit(
            patient_id=saved.patient_id,
            visit_type=saved.visit_type.lower(),
            case_text=saved.case_text,
            timestamp=saved.timestamp,
        )
        return saved

    def list_saved_results(self, patient_id: str) -> list[SavedCaseResult]:
        raw = patient_id.strip()
        if not raw:
            return []
        records = self.saved_result_store.list_records(raw)
        return list(reversed(records))

    def load_saved_result_by_timestamp(
        self,
        patient_id: str,
        timestamp: str,
    ) -> SavedCaseResult | None:
        raw_patient_id = patient_id.strip()
        raw_timestamp = timestamp.strip()
        if not raw_patient_id or not raw_timestamp:
            return None
        saved = self.saved_result_store.get_by_timestamp(raw_patient_id, raw_timestamp)
        if saved is None:
            return None
        self.visit_store.record_visit(
            patient_id=saved.patient_id,
            visit_type=saved.visit_type.lower(),
            case_text=saved.case_text,
            timestamp=saved.timestamp,
        )
        return saved

    def saved_case_to_pipeline_result(self, saved: SavedCaseResult) -> PipelineResult:
        return self._build_pipeline_result(
            patient_id=saved.patient_id,
            timestamp=saved.timestamp,
            case_text=saved.case_text,
            age=saved.age,
            weight=saved.weight,
            symptoms=saved.symptoms,
            spo2=saved.spo2,
            heart_rate=saved.heart_rate,
            history=saved.history,
            final_output=saved.final_opinion,
            selected_tools=saved.selected_tools,
            result_source="saved",
        )

    def process_visit(self, visit_input: PatientVisitInput) -> PipelineResult:
        self.ensure_runtime_ready()

        patient_id = visit_input.resolved_patient_id()
        timestamp = utcnow_iso()
        case_text = sanitize_provider_text(visit_input.to_case_text(patient_id))
        final_output, selected_tools = self._run_agentic(case_text, run_id=patient_id)

        self.visit_store.record_visit(
            patient_id=patient_id,
            visit_type=visit_input.normalized_visit_type(),
            case_text=case_text,
            timestamp=timestamp,
        )
        self.saved_result_store.save(
            SavedCaseResult(
                patient_id=patient_id,
                timestamp=timestamp,
                visit_type=visit_input.visit_type.strip().title(),
                age=visit_input.age,
                weight=visit_input.weight,
                symptoms=visit_input.symptoms.strip(),
                spo2=visit_input.spo2,
                heart_rate=visit_input.heart_rate,
                history=visit_input.history.strip(),
                case_text=case_text,
                final_opinion=final_output,
                selected_tools=selected_tools,
            )
        )
        return self._build_pipeline_result(
            patient_id=patient_id,
            timestamp=timestamp,
            case_text=case_text,
            age=visit_input.age,
            weight=visit_input.weight,
            symptoms=visit_input.symptoms.strip(),
            spo2=visit_input.spo2,
            heart_rate=visit_input.heart_rate,
            history=visit_input.history.strip(),
            final_output=final_output,
            selected_tools=selected_tools,
            result_source="live",
        )
