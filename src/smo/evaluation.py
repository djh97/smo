from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .pipeline import AgenticSMOService
from .reference_data import REFERENCE_ANSWERS, REFERENCE_REPEAT_ANSWER, REPEATABILITY_CASE, TEST_CASES


PRICING_PER_1M = {
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "claude-opus-4-6": {"in": 5.00, "out": 25.00},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
}

MODEL_LABEL = {
    "gpt-4o": "GPT-4o",
    "claude-opus-4-6": "Claude Opus 4.6",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}

TOOL_MODELS = ["gpt-4o", "claude-opus-4-6", "gemini-2.5-flash"]

COMPONENT_COLORS = {
    "GPT-4o Tool": "#4C78A8",
    "Claude Tool": "#F58518",
    "Gemini Tool": "#54A24B",
    "Controller": "#9C755F",
    "Guideline Retrieval": "#B8B0A2",
    "Full Agentic Pipeline": "#1B6E5A",
}


class AlignmentScorer:
    def __init__(self) -> None:
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def score(self, text_a: str, text_b: str) -> float:
        emb_a = self._model.encode(text_a, convert_to_numpy=True)
        emb_b = self._model.encode(text_b, convert_to_numpy=True)
        return self.cosine_similarity(emb_a, emb_b)


def build_case_prompt(case: dict[str, object]) -> str:
    return (
        f"Patient ID: {case['patient_id']}\n"
        f"Visit Type: {case['visit_type']}\n"
        f"Age: {case['age']}, Weight: {case['weight']} kg\n"
        f"Symptoms: {case['symptoms']}\n"
        f"SpO2: {case['spo2']}%, Heart Rate: {case['heart_rate']} bpm\n"
        f"Medical History: {case['history']}"
    )


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_alignment_experiment(service: AgenticSMOService) -> tuple[pd.DataFrame, pd.DataFrame]:
    scorer = AlignmentScorer()
    service.prompt_logger.activate_bucket("four_case")
    service.prompt_logger.clear_bucket("four_case")

    alignment_rows: list[dict[str, object]] = []
    for case in TEST_CASES:
        patient_id = str(case["patient_id"])
        case_prompt = build_case_prompt(case)
        reference = REFERENCE_ANSWERS[patient_id]

        gpt_base_out = service.openai_baseline_tool(case_prompt)
        claude_base_out = service.claude_baseline_tool(case_prompt)
        gemini_base_out = service.gemini_baseline_tool(case_prompt)

        previous_state = service.prompt_logger.enabled
        service.prompt_logger.enabled = False
        try:
            gpt_rag_out = service.openai_rag_tool(case_prompt, run_id=patient_id)
            claude_rag_out = service.claude_rag_tool(case_prompt, run_id=patient_id)
            gemini_rag_out = service.gemini_rag_tool(case_prompt, run_id=patient_id)
        finally:
            service.prompt_logger.enabled = previous_state

        agentic_out = service.run_agentic_combined_rag_synthesis(
            visit_type=str(case["visit_type"]),
            patient_id=patient_id,
            age=float(case["age"]),
            weight=float(case["weight"]),
            symptoms=str(case["symptoms"]),
            spo2=float(case["spo2"]),
            heart_rate=float(case["heart_rate"]),
            history=str(case["history"]),
        )

        alignment_rows.extend(
            [
                {
                    "Patient ID": patient_id,
                    "Case": patient_id,
                    "Model": "GPT-4o",
                    "Config": "Baseline",
                    "Alignment (cosine)": scorer.score(gpt_base_out, reference),
                },
                {
                    "Patient ID": patient_id,
                    "Case": patient_id,
                    "Model": "GPT-4o",
                    "Config": "RAG",
                    "Alignment (cosine)": scorer.score(gpt_rag_out, reference),
                },
                {
                    "Patient ID": patient_id,
                    "Case": patient_id,
                    "Model": "Claude Opus 4.6",
                    "Config": "Baseline",
                    "Alignment (cosine)": scorer.score(claude_base_out, reference),
                },
                {
                    "Patient ID": patient_id,
                    "Case": patient_id,
                    "Model": "Claude Opus 4.6",
                    "Config": "RAG",
                    "Alignment (cosine)": scorer.score(claude_rag_out, reference),
                },
                {
                    "Patient ID": patient_id,
                    "Case": patient_id,
                    "Model": "Gemini 2.5 Flash",
                    "Config": "Baseline",
                    "Alignment (cosine)": scorer.score(gemini_base_out, reference),
                },
                {
                    "Patient ID": patient_id,
                    "Case": patient_id,
                    "Model": "Gemini 2.5 Flash",
                    "Config": "RAG",
                    "Alignment (cosine)": scorer.score(gemini_rag_out, reference),
                },
                {
                    "Patient ID": patient_id,
                    "Case": patient_id,
                    "Model": "Agentic Pipeline",
                    "Config": "Agentic",
                    "Alignment (cosine)": scorer.score(agentic_out, reference),
                },
            ]
        )

    alignment_df = pd.DataFrame(alignment_rows)
    summary_df = (
        alignment_df.groupby(["Model", "Config"])["Alignment (cosine)"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "Mean Alignment", "std": "Std Alignment"})
    )
    return alignment_df, summary_df


def save_alignment_artifacts(
    alignment_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    ensure_output_dir(output_dir)
    alignment_df.to_csv(output_dir / "alignment_scores_per_case.csv", index=False)
    summary_df.to_csv(output_dir / "alignment_summary_baseline_rag_agentic.csv", index=False)

    df_plot = summary_df.copy()
    display_order = [
        ("GPT-4o", "Baseline"),
        ("GPT-4o", "RAG"),
        ("Claude Opus 4.6", "Baseline"),
        ("Claude Opus 4.6", "RAG"),
        ("Gemini 2.5 Flash", "Baseline"),
        ("Gemini 2.5 Flash", "RAG"),
        ("Agentic Pipeline", "Agentic"),
    ]
    display_labels = {
        ("GPT-4o", "Baseline"): "GPT-4o Baseline",
        ("GPT-4o", "RAG"): "GPT-4o RAG",
        ("Claude Opus 4.6", "Baseline"): "Claude Baseline",
        ("Claude Opus 4.6", "RAG"): "Claude RAG",
        ("Gemini 2.5 Flash", "Baseline"): "Gemini Baseline",
        ("Gemini 2.5 Flash", "RAG"): "Gemini RAG",
        ("Agentic Pipeline", "Agentic"): "Agentic Pipeline",
    }
    palette = {
        "Baseline": "#C7CEDB",
        "RAG": "#4C78A8",
        "Agentic": "#1B6E5A",
    }

    df_plot["order"] = df_plot.apply(
        lambda row: display_order.index((row["Model"], row["Config"])),
        axis=1,
    )
    df_plot["Display Label"] = df_plot.apply(
        lambda row: display_labels[(row["Model"], row["Config"])],
        axis=1,
    )
    df_plot["Color"] = df_plot["Config"].map(palette)
    df_plot = df_plot.sort_values("order").reset_index(drop=True)

    means = df_plot["Mean Alignment"].to_numpy(dtype=float)
    stds = df_plot["Std Alignment"].fillna(0.0).to_numpy(dtype=float)
    labels = df_plot["Display Label"].tolist()
    colors = df_plot["Color"].tolist()
    y_positions = np.arange(len(df_plot))[::-1]

    plt.figure(figsize=(9, 5.6))
    ax = plt.gca()
    for y, mean, std, color in zip(y_positions, means, stds, colors, strict=True):
        ax.hlines(y, 0, mean, color="#E6E9EF", linewidth=2, zorder=1)
        ax.errorbar(
            mean,
            y,
            xerr=std,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            markersize=7,
            zorder=3,
        )
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean Cosine Alignment")
    ax.set_title("Mean Cosine Alignment With Cross-Case Variability")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "alignment_std.pdf")
    plt.close()

    plt.figure(figsize=(9, 5.2))
    ax = plt.gca()
    ax.barh(y_positions, stds, color=colors, height=0.58)
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0.0, max(0.1, float(stds.max()) + 0.02))
    ax.set_xlabel("Standard Deviation Across Cases")
    ax.set_title("Cross-Case Alignment Variability by Configuration")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "alignment_variability.pdf")
    plt.close()


def run_repeatability_experiment(
    service: AgenticSMOService,
    *,
    n_runs: int = 30,
) -> tuple[list[float], pd.DataFrame]:
    scorer = AlignmentScorer()
    service.prompt_logger.activate_bucket("repeatability")
    service.prompt_logger.clear_bucket("repeatability")

    scores: list[float] = []
    for run_number in range(1, n_runs + 1):
        run_id = f"{REPEATABILITY_CASE['patient_id']}-run-{run_number:02d}"
        output = service.run_agentic_combined_rag_synthesis(
            visit_type=str(REPEATABILITY_CASE["visit_type"]),
            patient_id=str(REPEATABILITY_CASE["patient_id"]),
            age=float(REPEATABILITY_CASE["age"]),
            weight=float(REPEATABILITY_CASE["weight"]),
            symptoms=str(REPEATABILITY_CASE["symptoms"]),
            spo2=float(REPEATABILITY_CASE["spo2"]),
            heart_rate=float(REPEATABILITY_CASE["heart_rate"]),
            history=str(REPEATABILITY_CASE["history"]),
            run_id=run_id,
        )
        scores.append(scorer.score(output, REFERENCE_REPEAT_ANSWER))

    scores_arr = np.array(scores)
    se_alignment = scores_arr.std() / np.sqrt(n_runs)
    summary_df = pd.DataFrame(
        [
            {
                "n_runs": n_runs,
                "mean_alignment": float(scores_arr.mean()),
                "std_alignment": float(scores_arr.std()),
                "min_alignment": float(scores_arr.min()),
                "max_alignment": float(scores_arr.max()),
                "cv_alignment": float(scores_arr.std() / scores_arr.mean()),
                "ci_low_95": float(scores_arr.mean() - 1.96 * se_alignment),
                "ci_high_95": float(scores_arr.mean() + 1.96 * se_alignment),
            }
        ]
    )
    return scores, summary_df


def save_repeatability_artifacts(
    scores: list[float],
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    ensure_output_dir(output_dir)
    summary_df.to_csv(output_dir / "repeatability_30runs_summary.csv", index=False)

    scores_arr = np.array(scores)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.boxplot(scores_arr, vert=False)
    plt.title("Boxplot")
    plt.xlabel("Cosine Score")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(scores_arr) + 1), scores_arr, marker="o")
    plt.axhline(scores_arr.mean(), linestyle="--")
    plt.title("Run-to-Run Stability")
    plt.xlabel("Run")
    plt.ylabel("Cosine")
    plt.tight_layout()
    plt.savefig(output_dir / "repeatability_all.pdf")
    plt.close()


def cost_from_tokens(model: str, tokens_in: int, tokens_out: int) -> float:
    pricing = PRICING_PER_1M[model]
    return (tokens_in / 1_000_000) * pricing["in"] + (tokens_out / 1_000_000) * pricing["out"]


def build_dynamic_cost_breakdown_dataframe(
    prompt_log_bucket: dict[str, list[dict[str, object]]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id in sorted(prompt_log_bucket.keys()):
        calls = prompt_log_bucket[run_id]
        per_tool = {model: {"in": 0, "out": 0, "cost": 0.0} for model in TOOL_MODELS}
        controller = {"in": 0, "out": 0, "cost": 0.0}
        used_tools: set[str] = set()
        total_cost = 0.0

        for call in calls:
            model = str(call["model"])
            role = str(call.get("role", ""))
            tokens_in = int(call["tokens_in"])
            tokens_out = int(call["tokens_out"])

            if role == "tool" and model in per_tool:
                call_cost = cost_from_tokens(model, tokens_in, tokens_out)
                per_tool[model]["in"] += tokens_in
                per_tool[model]["out"] += tokens_out
                per_tool[model]["cost"] += call_cost
                total_cost += call_cost
                used_tools.add(model)
            elif role == "controller" and model == "gpt-4o":
                call_cost = cost_from_tokens(model, tokens_in, tokens_out)
                controller["in"] += tokens_in
                controller["out"] += tokens_out
                controller["cost"] += call_cost
                total_cost += call_cost

        rows.append(
            {
                "Case": run_id,
                "Tools Used": ", ".join(MODEL_LABEL[model] for model in sorted(used_tools)) if used_tools else "None",
                "GPT-4o Tool Input": per_tool["gpt-4o"]["in"],
                "GPT-4o Tool Output": per_tool["gpt-4o"]["out"],
                "GPT-4o Tool Cost (USD)": per_tool["gpt-4o"]["cost"],
                "Claude Tool Input (est)": per_tool["claude-opus-4-6"]["in"],
                "Claude Tool Output (est)": per_tool["claude-opus-4-6"]["out"],
                "Claude Tool Cost (USD)": per_tool["claude-opus-4-6"]["cost"],
                "Gemini Tool Input (est)": per_tool["gemini-2.5-flash"]["in"],
                "Gemini Tool Output (est)": per_tool["gemini-2.5-flash"]["out"],
                "Gemini Tool Cost (USD)": per_tool["gemini-2.5-flash"]["cost"],
                "Controller (GPT-4o) Input": controller["in"],
                "Controller (GPT-4o) Output": controller["out"],
                "Controller (GPT-4o) Cost (USD)": controller["cost"],
                "Total Cost (USD)": total_cost,
            }
        )

    dataframe = pd.DataFrame(rows)
    for column in [item for item in dataframe.columns if "Cost" in item]:
        dataframe[column] = dataframe[column].astype(float).round(6)
    return dataframe


def build_run_level_cost_dataframe(
    prompt_log_bucket: dict[str, list[dict[str, object]]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id in sorted(prompt_log_bucket.keys()):
        per_model: dict[str, dict[str, float]] = {}
        total = 0.0
        for call in prompt_log_bucket[run_id]:
            model = str(call["model"])
            tokens_in = int(call["tokens_in"])
            tokens_out = int(call["tokens_out"])
            if model not in PRICING_PER_1M:
                continue
            per_model.setdefault(model, {"in": 0, "out": 0, "cost": 0.0})
            per_model[model]["in"] += tokens_in
            per_model[model]["out"] += tokens_out
            model_cost = cost_from_tokens(model, tokens_in, tokens_out)
            per_model[model]["cost"] += model_cost
            total += model_cost

        rows.append(
            {
                "run_id": run_id,
                "gpt-4o_in": per_model.get("gpt-4o", {}).get("in", 0),
                "gpt-4o_out": per_model.get("gpt-4o", {}).get("out", 0),
                "claude_in_est": per_model.get("claude-opus-4-6", {}).get("in", 0),
                "claude_out_est": per_model.get("claude-opus-4-6", {}).get("out", 0),
                "gemini_in_est": per_model.get("gemini-2.5-flash", {}).get("in", 0),
                "gemini_out_est": per_model.get("gemini-2.5-flash", {}).get("out", 0),
                "total_cost": round(total, 6),
            }
        )
    return pd.DataFrame(rows)


def save_cost_artifacts(dataframe: pd.DataFrame, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    dataframe.to_csv(output_dir / "table9_dynamic_cost_breakdown.csv", index=False)

    avg_gpt_tool = float(dataframe["GPT-4o Tool Cost (USD)"].mean())
    avg_claude = float(dataframe["Claude Tool Cost (USD)"].mean())
    avg_gemini = float(dataframe["Gemini Tool Cost (USD)"].mean())
    avg_controller = float(dataframe["Controller (GPT-4o) Cost (USD)"].mean())
    avg_total = float(dataframe["Total Cost (USD)"].mean())

    labels = ["GPT-4o Tool", "Claude Tool", "Gemini Tool", "Controller", "Full Agentic Pipeline"]
    values = [avg_gpt_tool, avg_claude, avg_gemini, avg_controller, avg_total]

    colors = [COMPONENT_COLORS[label] for label in labels]
    y_positions = np.arange(len(labels))[::-1]

    plt.figure(figsize=(8.8, 4.8))
    ax = plt.gca()
    ax.barh(y_positions, values, color=colors, height=0.6)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Average Cost (USD)")
    ax.set_title("Average Cost per Agentic Component")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    for y, value in zip(y_positions, values, strict=True):
        ax.text(value + 0.0006, y, f"${value:.4f}", va="center", ha="left", fontsize=9)
    ax.set_xlim(0.0, max(values) * 1.22)
    plt.tight_layout()
    plt.savefig(output_dir / "agentic_cost_avg_dynamic.png", dpi=300)
    plt.close()


def build_dynamic_latency_breakdown_dataframe(
    prompt_log_bucket: dict[str, list[dict[str, object]]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id in sorted(prompt_log_bucket.keys()):
        calls = prompt_log_bucket[run_id]
        retrieval_time = 0.0
        gpt_tool_time = 0.0
        claude_tool_time = 0.0
        gemini_tool_time = 0.0
        controller_time = 0.0
        total_time = 0.0
        used_tools: set[str] = set()

        for call in calls:
            model = str(call.get("model", ""))
            role = str(call.get("role", ""))
            duration = float(call.get("duration_seconds", 0.0) or 0.0)

            if role == "retrieval":
                retrieval_time += duration
            elif role == "tool" and model == "gpt-4o":
                gpt_tool_time += duration
                used_tools.add(model)
            elif role == "tool" and model == "claude-opus-4-6":
                claude_tool_time += duration
                used_tools.add(model)
            elif role == "tool" and model == "gemini-2.5-flash":
                gemini_tool_time += duration
                used_tools.add(model)
            elif role == "controller" and model == "gpt-4o":
                controller_time += duration
            elif role == "total" and model == "pipeline":
                total_time += duration

        if total_time <= 0.0:
            total_time = retrieval_time + gpt_tool_time + claude_tool_time + gemini_tool_time + controller_time

        rows.append(
            {
                "Case": run_id,
                "Tools Used": ", ".join(MODEL_LABEL[model] for model in sorted(used_tools)) if used_tools else "None",
                "Guideline Retrieval (s)": retrieval_time,
                "GPT-4o Tool (s)": gpt_tool_time,
                "Claude Tool (s)": claude_tool_time,
                "Gemini Tool (s)": gemini_tool_time,
                "Controller (s)": controller_time,
                "Total Time (s)": total_time,
            }
        )

    dataframe = pd.DataFrame(rows)
    time_columns = [column for column in dataframe.columns if column.endswith("(s)")]
    for column in time_columns:
        dataframe[column] = dataframe[column].astype(float).round(2)
    return dataframe


def save_latency_artifacts(dataframe: pd.DataFrame, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    dataframe.to_csv(output_dir / "table10_dynamic_latency_breakdown.csv", index=False)

    labels = [
        "Guideline Retrieval",
        "GPT-4o Tool",
        "Claude Tool",
        "Gemini Tool",
        "Controller",
        "Full Agentic Pipeline",
    ]
    values = [
        float(dataframe["Guideline Retrieval (s)"].mean()),
        float(dataframe["GPT-4o Tool (s)"].mean()),
        float(dataframe["Claude Tool (s)"].mean()),
        float(dataframe["Gemini Tool (s)"].mean()),
        float(dataframe["Controller (s)"].mean()),
        float(dataframe["Total Time (s)"].mean()),
    ]
    colors = [COMPONENT_COLORS[label] for label in labels]
    y_positions = np.arange(len(labels))[::-1]

    plt.figure(figsize=(8.8, 5.0))
    ax = plt.gca()
    ax.barh(y_positions, values, color=colors, height=0.6)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Average Time (s)")
    ax.set_title("Average Latency per Agentic Component")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    for y, value in zip(y_positions, values, strict=True):
        ax.text(value + 0.12, y, f"{value:.2f}s", va="center", ha="left", fontsize=9)
    ax.set_xlim(0.0, max(values) * 1.22 if values else 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "av_time.png", dpi=300)
    plt.close()


def save_prompt_log_json(
    prompt_log_bucket: dict[str, list[dict[str, object]]],
    output_path: Path,
) -> None:
    ensure_output_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(prompt_log_bucket, handle, indent=2)
