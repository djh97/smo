from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment-dependent
    pd = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

try:
    from smo.geval import (
        build_geval_metrics_dataframe,
        build_threshold_selection_summary,
        build_threshold_sweep_dataframe,
        choose_operating_threshold,
        classify_geval_score,
    )
except ImportError as exc:  # pragma: no cover - environment-dependent
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"G-Eval test dependencies are unavailable: {IMPORT_ERROR}")
class GEvalMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scores_df = pd.DataFrame(
            [
                {"Model": "GPT-4o", "Ground Truth": "Positive", "Score": 0.90},
                {"Model": "GPT-4o", "Ground Truth": "Positive", "Score": 0.90},
                {"Model": "GPT-4o", "Ground Truth": "Positive", "Score": 0.90},
                {"Model": "GPT-4o", "Ground Truth": "Positive", "Score": 0.87},
                {"Model": "GPT-4o", "Ground Truth": "Negative", "Score": 0.20},
                {"Model": "GPT-4o", "Ground Truth": "Negative", "Score": 0.15},
                {"Model": "GPT-4o", "Ground Truth": "Negative", "Score": 0.10},
                {"Model": "GPT-4o", "Ground Truth": "Negative", "Score": 0.30},
                {"Model": "Claude Opus 4.6", "Ground Truth": "Positive", "Score": 0.90},
                {"Model": "Claude Opus 4.6", "Ground Truth": "Positive", "Score": 0.89},
                {"Model": "Claude Opus 4.6", "Ground Truth": "Positive", "Score": 0.65},
                {"Model": "Claude Opus 4.6", "Ground Truth": "Positive", "Score": 0.87},
                {"Model": "Claude Opus 4.6", "Ground Truth": "Negative", "Score": 0.20},
                {"Model": "Claude Opus 4.6", "Ground Truth": "Negative", "Score": 0.18},
                {"Model": "Claude Opus 4.6", "Ground Truth": "Negative", "Score": 0.32},
                {"Model": "Claude Opus 4.6", "Ground Truth": "Negative", "Score": 0.25},
                {"Model": "Gemini 2.5 Flash", "Ground Truth": "Positive", "Score": 0.90},
                {"Model": "Gemini 2.5 Flash", "Ground Truth": "Positive", "Score": 0.89},
                {"Model": "Gemini 2.5 Flash", "Ground Truth": "Positive", "Score": 0.65},
                {"Model": "Gemini 2.5 Flash", "Ground Truth": "Positive", "Score": 0.87},
                {"Model": "Gemini 2.5 Flash", "Ground Truth": "Negative", "Score": 0.22},
                {"Model": "Gemini 2.5 Flash", "Ground Truth": "Negative", "Score": 0.21},
                {"Model": "Gemini 2.5 Flash", "Ground Truth": "Negative", "Score": 0.19},
                {"Model": "Gemini 2.5 Flash", "Ground Truth": "Negative", "Score": 0.35},
                {"Model": "Agentic", "Ground Truth": "Positive", "Score": 0.92},
                {"Model": "Agentic", "Ground Truth": "Positive", "Score": 0.91},
                {"Model": "Agentic", "Ground Truth": "Positive", "Score": 0.92},
                {"Model": "Agentic", "Ground Truth": "Positive", "Score": 0.93},
                {"Model": "Agentic", "Ground Truth": "Negative", "Score": 0.12},
                {"Model": "Agentic", "Ground Truth": "Negative", "Score": 0.10},
                {"Model": "Agentic", "Ground Truth": "Negative", "Score": 0.08},
                {"Model": "Agentic", "Ground Truth": "Negative", "Score": 0.14},
            ]
        )

    def test_classify_geval_score(self) -> None:
        self.assertEqual(classify_geval_score(0.89, 0.89), "Positive")
        self.assertEqual(classify_geval_score(0.88, 0.89), "Negative")

    def test_build_geval_metrics_dataframe_matches_binary_classification_logic(self) -> None:
        metrics_df = build_geval_metrics_dataframe(self.scores_df, threshold=0.89)
        records = {row["Model"]: row for row in metrics_df.to_dict(orient="records")}

        self.assertEqual(records["GPT-4o"]["TP"], 3)
        self.assertEqual(records["GPT-4o"]["FP"], 0)
        self.assertEqual(records["GPT-4o"]["TN"], 4)
        self.assertEqual(records["GPT-4o"]["FN"], 1)
        self.assertEqual(records["GPT-4o"]["Precision"], 1.00)
        self.assertEqual(records["GPT-4o"]["Recall"], 0.75)
        self.assertEqual(records["GPT-4o"]["Specificity"], 1.00)
        self.assertEqual(records["GPT-4o"]["F1-score"], 0.86)

        self.assertEqual(records["Claude Opus 4.6"]["TP"], 2)
        self.assertEqual(records["Claude Opus 4.6"]["FP"], 0)
        self.assertEqual(records["Claude Opus 4.6"]["TN"], 4)
        self.assertEqual(records["Claude Opus 4.6"]["FN"], 2)
        self.assertEqual(records["Claude Opus 4.6"]["Precision"], 1.00)
        self.assertEqual(records["Claude Opus 4.6"]["Recall"], 0.50)
        self.assertEqual(records["Claude Opus 4.6"]["Specificity"], 1.00)
        self.assertEqual(records["Claude Opus 4.6"]["F1-score"], 0.67)

        self.assertEqual(records["Gemini 2.5 Flash"]["TP"], 2)
        self.assertEqual(records["Gemini 2.5 Flash"]["FP"], 0)
        self.assertEqual(records["Gemini 2.5 Flash"]["TN"], 4)
        self.assertEqual(records["Gemini 2.5 Flash"]["FN"], 2)
        self.assertEqual(records["Gemini 2.5 Flash"]["Precision"], 1.00)
        self.assertEqual(records["Gemini 2.5 Flash"]["Recall"], 0.50)
        self.assertEqual(records["Gemini 2.5 Flash"]["Specificity"], 1.00)
        self.assertEqual(records["Gemini 2.5 Flash"]["F1-score"], 0.67)

        self.assertEqual(records["Agentic"]["TP"], 4)
        self.assertEqual(records["Agentic"]["FP"], 0)
        self.assertEqual(records["Agentic"]["TN"], 4)
        self.assertEqual(records["Agentic"]["FN"], 0)
        self.assertEqual(records["Agentic"]["Precision"], 1.00)
        self.assertEqual(records["Agentic"]["Recall"], 1.00)
        self.assertEqual(records["Agentic"]["Specificity"], 1.00)
        self.assertEqual(records["Agentic"]["F1-score"], 1.00)

    def test_build_threshold_sweep_dataframe(self) -> None:
        threshold_df = build_threshold_sweep_dataframe(self.scores_df, thresholds=[0.89, 0.95])
        records = threshold_df.to_dict(orient="records")

        self.assertEqual(records[0]["True Positives"], 11)
        self.assertEqual(records[0]["False Positives"], 0)
        self.assertEqual(records[0]["True Negatives"], 16)
        self.assertEqual(records[0]["False Negatives"], 5)
        self.assertEqual(records[1]["True Positives"], 0)
        self.assertEqual(records[1]["False Positives"], 0)
        self.assertEqual(records[1]["True Negatives"], 16)
        self.assertEqual(records[1]["False Negatives"], 16)

    def test_choose_operating_threshold_prefers_specificity_rule(self) -> None:
        selection_df = pd.DataFrame(
            [
                {"Selection Rule": "Best balanced accuracy", "Threshold": 0.40},
                {"Selection Rule": "Best F1-score", "Threshold": 0.40},
                {"Selection Rule": "Highest recall with FP=0", "Threshold": 0.95},
                {
                    "Selection Rule": "Best balanced accuracy with specificity>=0.90",
                    "Threshold": 0.89,
                },
            ]
        )
        threshold, rule = choose_operating_threshold(
            selection_df,
            preferred_rule="Best balanced accuracy with specificity>=0.90",
            fallback_threshold=0.89,
        )
        self.assertEqual(rule, "Best balanced accuracy with specificity>=0.90")
        self.assertEqual(threshold, 0.89)


if __name__ == "__main__":
    unittest.main()
