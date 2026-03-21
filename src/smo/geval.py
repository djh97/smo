from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from .reference_data import REFERENCE_ANSWERS, TEST_CASES
from .schemas import PromptLogEntry
from .text_safety import sanitize_provider_text

if TYPE_CHECKING:
    from .config import Settings
    from .pipeline import AgenticSMOService


G_EVAL_MODEL_ORDER = [
    "GPT-4o",
    "Claude Opus 4.6",
    "Gemini 2.5 Flash",
    "Agentic",
]

DEFAULT_THRESHOLD_SELECTION_RULE = "Manual fixed threshold"

G_EVAL_CASE_LABELS = {
    "P001": "Severe asthma",
    "P002": "Severe pneumonia",
    "P003": "Mild asthma",
    "P004": "Elderly CAP",
}

PAIR_TYPE_ORDER = ["Positive", "Near Negative", "Far Negative"]

# "Near" negatives stay within the same disease family when possible, while
# "Far" negatives use a clearly different respiratory context.
NEGATIVE_REFERENCE_MAP = {
    "P001": {"near": "P003", "far": "P004"},
    "P002": {"near": "P004", "far": "P003"},
    "P003": {"near": "P001", "far": "P004"},
    "P004": {"near": "P002", "far": "P003"},
}

CONFUSION_MATRIX_FILENAMES = {
    "GPT-4o": "gptmatrix.png",
    "Claude Opus 4.6": "claudematrix.png",
    "Gemini 2.5 Flash": "geminimatrix.png",
    "Agentic": "agenticmatrix.png",
}


@dataclass
class GEvalJudgment:
    score: float
    reason: str
    raw_response: str
    prompt: str


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


def classify_geval_score(score: float, threshold: float) -> str:
    return "Positive" if float(score) >= float(threshold) else "Negative"


def build_reference_pairs(case_id: str) -> list[dict[str, str]]:
    mapping = NEGATIVE_REFERENCE_MAP[case_id]
    return [
        {
            "Reference Patient ID": case_id,
            "Reference Clinical Case": G_EVAL_CASE_LABELS[case_id],
            "Pair Type": "Positive",
            "Ground Truth": "Positive",
        },
        {
            "Reference Patient ID": mapping["near"],
            "Reference Clinical Case": G_EVAL_CASE_LABELS[mapping["near"]],
            "Pair Type": "Near Negative",
            "Ground Truth": "Negative",
        },
        {
            "Reference Patient ID": mapping["far"],
            "Reference Clinical Case": G_EVAL_CASE_LABELS[mapping["far"]],
            "Pair Type": "Far Negative",
            "Ground Truth": "Negative",
        },
    ]


def _classification_metrics(tp: int, fp: int, tn: int, fn: int) -> tuple[float, float, float, float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    balanced_accuracy = (recall + specificity) / 2 if (tp + fp + tn + fn) else 0.0
    return precision, recall, specificity, f1, accuracy, balanced_accuracy


def build_geval_metrics_dataframe(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in G_EVAL_MODEL_ORDER:
        model_df = scores_df[scores_df["Model"] == model]
        if model_df.empty:
            continue

        y_true = model_df["Ground Truth"].to_numpy()
        y_pred = model_df["Score"].apply(lambda value: classify_geval_score(value, threshold)).to_numpy()

        tp = int(np.sum((y_true == "Positive") & (y_pred == "Positive")))
        fp = int(np.sum((y_true == "Negative") & (y_pred == "Positive")))
        tn = int(np.sum((y_true == "Negative") & (y_pred == "Negative")))
        fn = int(np.sum((y_true == "Positive") & (y_pred == "Negative")))
        precision, recall, specificity, f1, accuracy, balanced_accuracy = _classification_metrics(
            tp, fp, tn, fn
        )

        rows.append(
            {
                "Model": model,
                "Threshold": float(threshold),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "Precision": round(precision, 2),
                "Recall": round(recall, 2),
                "Specificity": round(specificity, 2),
                "F1-score": round(f1, 2),
                "Accuracy": round(accuracy, 2),
                "Balanced Accuracy": round(balanced_accuracy, 2),
            }
        )

    return pd.DataFrame(rows)


def build_threshold_sweep_dataframe(
    scores_df: pd.DataFrame,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    y_true = scores_df["Ground Truth"].to_numpy()
    scores = scores_df["Score"].astype(float).to_numpy()
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        y_pred = np.where(scores >= float(threshold), "Positive", "Negative")
        tp = int(np.sum((y_true == "Positive") & (y_pred == "Positive")))
        fp = int(np.sum((y_true == "Negative") & (y_pred == "Positive")))
        tn = int(np.sum((y_true == "Negative") & (y_pred == "Negative")))
        fn = int(np.sum((y_true == "Positive") & (y_pred == "Negative")))
        precision, recall, specificity, f1, accuracy, balanced_accuracy = _classification_metrics(
            tp, fp, tn, fn
        )
        rows.append(
            {
                "Threshold": float(threshold),
                "True Positives": tp,
                "False Positives": fp,
                "True Negatives": tn,
                "False Negatives": fn,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "Specificity": round(specificity, 4),
                "F1-score": round(f1, 4),
                "Accuracy": round(accuracy, 4),
                "Balanced Accuracy": round(balanced_accuracy, 4),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_selection_summary(threshold_df: pd.DataFrame) -> pd.DataFrame:
    if threshold_df.empty:
        return pd.DataFrame(
            columns=[
                "Selection Rule",
                "Threshold",
                "True Positives",
                "False Positives",
                "True Negatives",
                "False Negatives",
                "Precision",
                "Recall",
                "Specificity",
                "F1-score",
                "Accuracy",
                "Balanced Accuracy",
            ]
        )

    rows: list[dict[str, object]] = []

    def select_row(label: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        row = df.iloc[0].to_dict()
        rows.append({"Selection Rule": label, **row})

    best_balanced = threshold_df.sort_values(
        ["Balanced Accuracy", "F1-score", "Specificity", "Threshold"],
        ascending=[False, False, False, True],
    )
    select_row("Best balanced accuracy", best_balanced)

    best_f1 = threshold_df.sort_values(
        ["F1-score", "Balanced Accuracy", "Specificity", "Threshold"],
        ascending=[False, False, False, True],
    )
    select_row("Best F1-score", best_f1)

    fp_zero = threshold_df[threshold_df["False Positives"] == 0].sort_values(
        ["Recall", "Balanced Accuracy", "Threshold"],
        ascending=[False, False, True],
    )
    select_row("Highest recall with FP=0", fp_zero)

    specificity_90 = threshold_df[threshold_df["Specificity"] >= 0.90].sort_values(
        ["Balanced Accuracy", "Recall", "Threshold"],
        ascending=[False, False, True],
    )
    select_row("Best balanced accuracy with specificity>=0.90", specificity_90)

    return pd.DataFrame(rows)


def choose_operating_threshold(
    threshold_selection_df: pd.DataFrame,
    preferred_rule: str = DEFAULT_THRESHOLD_SELECTION_RULE,
    fallback_threshold: float = 0.89,
) -> tuple[float, str]:
    if preferred_rule == "Manual fixed threshold":
        return float(fallback_threshold), preferred_rule

    if not threshold_selection_df.empty:
        preferred = threshold_selection_df[
            threshold_selection_df["Selection Rule"] == preferred_rule
        ]
        if not preferred.empty:
            return float(preferred.iloc[0]["Threshold"]), preferred_rule

        first = threshold_selection_df.iloc[0]
        return float(first["Threshold"]), str(first["Selection Rule"])

    return float(fallback_threshold), "Fallback threshold"


def build_geval_model_summary_dataframe(scores_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in G_EVAL_MODEL_ORDER:
        model_df = scores_df[scores_df["Model"] == model]
        if model_df.empty:
            continue

        metric_row = metrics_df[metrics_df["Model"] == model].iloc[0].to_dict()

        def mean_for(pair_type: str) -> float:
            subset = model_df[model_df["Pair Type"] == pair_type]["Score"]
            return round(float(subset.mean()), 4) if not subset.empty else float("nan")

        rows.append(
            {
                "Model": model,
                "Mean Positive Score": mean_for("Positive"),
                "Mean Near-Negative Score": mean_for("Near Negative"),
                "Mean Far-Negative Score": mean_for("Far Negative"),
                "Threshold": metric_row["Threshold"],
                "TP": metric_row["TP"],
                "FP": metric_row["FP"],
                "TN": metric_row["TN"],
                "FN": metric_row["FN"],
                "Precision": metric_row["Precision"],
                "Recall": metric_row["Recall"],
                "Specificity": metric_row["Specificity"],
                "F1-score": metric_row["F1-score"],
                "Accuracy": metric_row["Accuracy"],
                "Balanced Accuracy": metric_row["Balanced Accuracy"],
            }
        )

    return pd.DataFrame(rows)


def _extract_json_object(raw_response: str) -> dict[str, object]:
    sanitized = sanitize_provider_text(raw_response).strip()
    sanitized = re.sub(r"^```(?:json)?\s*", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s*```$", "", sanitized)
    match = re.search(r"\{.*\}", sanitized, flags=re.DOTALL)
    candidate = match.group(0) if match else sanitized
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse judge JSON response: {raw_response}") from exc


class GEvalJudge:
    def __init__(self, service: AgenticSMOService) -> None:
        self.service = service
        self.settings: Settings = service.settings
        self._judge_llm = None

    def _get_llm(self):
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the G-Eval judge.")
        if self._judge_llm is None:
            from langchain_openai import ChatOpenAI

            self._judge_llm = ChatOpenAI(
                model=self.settings.geval_judge_model,
                temperature=0.0,
                api_key=self.settings.openai_api_key,
            )
        return self._judge_llm

    def evaluate(
        self,
        *,
        input_text: str,
        actual_output: str,
        expected_output: str,
        run_id: str,
        evaluated_model: str,
        pair_type: str,
        reference_case_id: str,
    ) -> GEvalJudgment:
        prompt = f"""
You are an expert medical evaluation judge.
Compare the ACTUAL system output against the EXPECTED WHO/MSF-aligned reference answer for the same patient case.

Evaluate only these dimensions:
- diagnostic correctness
- severity classification
- treatment completeness
- escalation or hospitalization appropriateness
- monitoring and follow-up appropriateness
- patient-population match, including age group, major risk context, and case-specific management details

Use a continuous score from 0.00 to 1.00:
- 1.00 = complete agreement with the reference standard
- 0.90 to 0.99 = strong agreement with only minor omissions or harmless wording differences
- 0.70 to 0.89 = partially aligned but missing clinically important details
- 0.40 to 0.69 = materially incomplete or clinically divergent
- 0.00 to 0.39 = poor alignment with the reference

Ignore formatting differences. Focus on medical reasoning and management completeness.

Important judging rules:
- Treat age group and patient population as clinically important, not cosmetic. A pediatric recommendation is not strongly aligned with an elderly-adult reference, and an elderly-adult recommendation is not strongly aligned with a pediatric reference, even if both concern the same disease family.
- Do not assign a high score merely because two outputs share generic management elements such as oxygen, antibiotics, hospitalization, or monitoring.
- If the ACTUAL output fits the wrong patient population, wrong severity context, or wrong case-specific management pathway, reduce the score materially.
- To deserve a score >= 0.90, the ACTUAL output should match the reference not only on disease family, but also on the patient-specific context and the key management details that follow from that context.

Return ONLY valid JSON with this schema:
{{"score": 0.0, "reason": "short explanation"}}

== Patient Case ==
{input_text}

== Expected Reference Answer ==
{expected_output}

== Actual System Output ==
{actual_output}
""".strip()
        raw_response = self.service._invoke(self._get_llm(), prompt)
        parsed = _extract_json_object(raw_response)

        score = float(parsed["score"])
        if 1.0 < score <= 5.0:
            score = score / 5.0
        if score < 0.0 or score > 1.0:
            raise RuntimeError(f"G-Eval score is outside the expected [0, 1] range: {score}")

        reason = sanitize_provider_text(str(parsed.get("reason", "")).strip())
        self.service.prompt_logger.log(
            PromptLogEntry(
                run_id=run_id,
                model=self.settings.geval_judge_model,
                role="judge",
                tokens_in=self.service._tokcount_openai(prompt, self.settings.geval_judge_model),
                tokens_out=self.service._tokcount_openai(raw_response, self.settings.geval_judge_model),
                note=f"G-Eval judgment for {evaluated_model} against {reference_case_id} ({pair_type})",
            )
        )
        return GEvalJudgment(
            score=round(score, 4),
            reason=reason,
            raw_response=raw_response,
            prompt=prompt,
        )


def ensure_geval_runtime_ready(service: AgenticSMOService) -> None:
    service.ensure_runtime_ready()
    if not service.settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for the manuscript-aligned G-Eval run.")
    if not service.settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for the manuscript-aligned G-Eval run.")


def run_geval_experiment(
    service: AgenticSMOService,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]], pd.DataFrame, float, str]:
    ensure_geval_runtime_ready(service)
    service.prompt_logger.activate_bucket("geval")
    service.prompt_logger.clear_bucket("geval")

    judge = GEvalJudge(service)
    rows: list[dict[str, object]] = []
    detailed_judgments: list[dict[str, object]] = []

    for case in TEST_CASES:
        patient_id = str(case["patient_id"])
        case_label = G_EVAL_CASE_LABELS.get(patient_id, patient_id)
        input_text = build_case_prompt(case)

        gpt_output = service.openai_rag_tool(input_text, run_id=patient_id)
        claude_output = service.claude_rag_tool(input_text, run_id=patient_id)
        gemini_output = service.gemini_rag_tool(input_text, run_id=patient_id)
        agentic_output = service.run_agentic_combined_rag_synthesis(
            visit_type=str(case["visit_type"]),
            patient_id=patient_id,
            age=float(case["age"]),
            weight=float(case["weight"]),
            symptoms=str(case["symptoms"]),
            spo2=float(case["spo2"]),
            heart_rate=float(case["heart_rate"]),
            history=str(case["history"]),
            run_id=patient_id,
        )

        outputs = {
            "GPT-4o": gpt_output,
            "Claude Opus 4.6": claude_output,
            "Gemini 2.5 Flash": gemini_output,
            "Agentic": agentic_output,
        }

        for pair in build_reference_pairs(patient_id):
            reference_case_id = pair["Reference Patient ID"]
            expected_output = REFERENCE_ANSWERS[reference_case_id]

            for model_name in G_EVAL_MODEL_ORDER:
                actual_output = outputs[model_name]
                judgment = judge.evaluate(
                    input_text=input_text,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    run_id=patient_id,
                    evaluated_model=model_name,
                    pair_type=pair["Pair Type"],
                    reference_case_id=reference_case_id,
                )
                row = {
                    "Patient ID": patient_id,
                    "Case": patient_id,
                    "Clinical Case": case_label,
                    "Reference Patient ID": reference_case_id,
                    "Reference Clinical Case": pair["Reference Clinical Case"],
                    "Pair Type": pair["Pair Type"],
                    "Ground Truth": pair["Ground Truth"],
                    "Model": model_name,
                    "Score": judgment.score,
                    "Evaluation Summary": judgment.reason,
                }
                rows.append(row)
                detailed_judgments.append(
                    {
                        **row,
                        "Judge Model": service.settings.geval_judge_model,
                        "Input": input_text,
                        "Expected Output": expected_output,
                        "Actual Output": actual_output,
                        "Judge Prompt": judgment.prompt,
                        "Judge Raw Response": judgment.raw_response,
                    }
                )

    scores_df = pd.DataFrame(rows)
    thresholds = np.linspace(0.0, 1.0, service.settings.geval_threshold_points)
    threshold_df = build_threshold_sweep_dataframe(scores_df, thresholds)
    threshold_selection_df = build_threshold_selection_summary(threshold_df)
    operating_threshold, threshold_rule = choose_operating_threshold(
        threshold_selection_df,
        preferred_rule=getattr(service.settings, "geval_threshold_rule", DEFAULT_THRESHOLD_SELECTION_RULE),
        fallback_threshold=service.settings.geval_threshold,
    )
    metrics_df = build_geval_metrics_dataframe(scores_df, operating_threshold)
    return (
        scores_df,
        metrics_df,
        threshold_df,
        detailed_judgments,
        threshold_selection_df,
        operating_threshold,
        threshold_rule,
    )


def save_geval_artifacts(
    scores_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    detailed_judgments: list[dict[str, object]],
    output_dir: Path,
    *,
    threshold: float,
    threshold_rule: str,
    threshold_selection_df: pd.DataFrame | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    ensure_output_dir(output_dir)

    ordered_scores = scores_df.copy()
    ordered_scores["Model"] = pd.Categorical(
        ordered_scores["Model"],
        categories=G_EVAL_MODEL_ORDER,
        ordered=True,
    )
    ordered_scores["Pair Type"] = pd.Categorical(
        ordered_scores["Pair Type"],
        categories=PAIR_TYPE_ORDER,
        ordered=True,
    )
    ordered_scores = ordered_scores.sort_values(["Patient ID", "Pair Type", "Model"]).reset_index(drop=True)

    positive_scores = ordered_scores[ordered_scores["Pair Type"] == "Positive"].copy()
    positive_matrix = (
        positive_scores.pivot(index="Clinical Case", columns="Model", values="Score")
        .reindex(columns=G_EVAL_MODEL_ORDER)
        .reset_index()
    )

    summary_df = build_geval_model_summary_dataframe(ordered_scores, metrics_df)
    threshold_selection_df = (
        threshold_selection_df
        if threshold_selection_df is not None
        else build_threshold_selection_summary(threshold_df)
    )
    selected_threshold_payload = {
        "selection_rule": threshold_rule,
        "selected_threshold": float(threshold),
    }

    ordered_scores.to_csv(output_dir / "g_eval_scores.csv", index=False)
    ordered_scores.to_csv(output_dir / "geval_pair_scores.csv", index=False)
    positive_scores.to_csv(output_dir / "geval_case_scores.csv", index=False)
    positive_matrix.to_csv(output_dir / "geval_positive_score_matrix.csv", index=False)
    summary_df.to_csv(output_dir / "geval_model_summary.csv", index=False)
    metrics_df.to_csv(output_dir / "llm_reference_precision_recall_f1.csv", index=False)
    threshold_df.to_csv(output_dir / "threshold_sweep.csv", index=False)
    threshold_selection_df.to_csv(output_dir / "threshold_selection_summary.csv", index=False)
    (output_dir / "selected_threshold.json").write_text(
        json.dumps(selected_threshold_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "geval_judgments.json").write_text(
        json.dumps(detailed_judgments, indent=2),
        encoding="utf-8",
    )

    pivot = positive_scores.pivot(index="Model", columns="Case", values="Score").reindex(G_EVAL_MODEL_ORDER)

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(pivot.columns))
    bar_width = 0.18
    for offset, model in enumerate(G_EVAL_MODEL_ORDER):
        values = pivot.loc[model].to_numpy(dtype=float)
        plt.bar(
            x_positions + (offset - 1.5) * bar_width,
            values,
            width=bar_width,
            label=model,
        )
    plt.xticks(x_positions, pivot.columns.tolist())
    plt.ylim(0, 1.05)
    plt.xlabel("Case")
    plt.ylabel("Positive-Pair G-Eval Score")
    plt.title("Positive-Pair G-Eval Scores by Model and Case")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "geval_scores_by_case.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    heatmap_values = pivot.to_numpy(dtype=float)
    image = plt.imshow(heatmap_values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(image, label="Score")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns.tolist())
    plt.yticks(np.arange(len(G_EVAL_MODEL_ORDER)), G_EVAL_MODEL_ORDER)
    for row_index in range(heatmap_values.shape[0]):
        for col_index in range(heatmap_values.shape[1]):
            plt.text(
                col_index,
                row_index,
                f"{heatmap_values[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    plt.title("Positive-Pair G-Eval Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "geval_heatmap.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["True Positives"],
        label="True positives",
        color="purple",
        linewidth=2,
    )
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["False Positives"],
        label="False positives",
        color="orange",
        linewidth=2,
    )
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["True Negatives"],
        label="True negatives",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["False Negatives"],
        label="False negatives",
        color="green",
        linewidth=2,
    )
    plt.axvline(float(threshold), linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Decision Threshold")
    plt.ylabel("Count")
    plt.title("Confusion matrix elements across threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "threshold.png", dpi=300)
    plt.close()

    for model_name, filename in CONFUSION_MATRIX_FILENAMES.items():
        model_df = ordered_scores[ordered_scores["Model"] == model_name]
        y_true = model_df["Ground Truth"].to_list()
        y_pred = [classify_geval_score(value, threshold) for value in model_df["Score"]]
        matrix = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative"])

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Positive", "Negative"],
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        display.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix: {model_name}")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)
