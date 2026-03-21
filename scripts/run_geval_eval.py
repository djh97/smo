from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smo.config import get_settings
from smo.evaluation import save_prompt_log_json
from smo.geval import run_geval_experiment, save_geval_artifacts
from smo.pipeline import AgenticSMOService


def main() -> None:
    settings = get_settings()
    service = AgenticSMOService(settings)
    output_dir = settings.output_dir

    (
        scores_df,
        metrics_df,
        threshold_df,
        detailed_judgments,
        threshold_selection_df,
        operating_threshold,
        threshold_rule,
    ) = run_geval_experiment(service)
    save_geval_artifacts(
        scores_df,
        metrics_df,
        threshold_df,
        detailed_judgments,
        output_dir,
        threshold=operating_threshold,
        threshold_rule=threshold_rule,
        threshold_selection_df=threshold_selection_df,
    )
    save_prompt_log_json(service.prompt_logger.bucket("geval"), output_dir / "prompt_log_geval.json")

    print(f"Saved G-Eval outputs to {output_dir}")


if __name__ == "__main__":
    main()
