from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smo.config import get_settings
from smo.evaluation import (
    build_run_level_cost_dataframe,
    run_repeatability_experiment,
    save_prompt_log_json,
    save_repeatability_artifacts,
)
from smo.pipeline import AgenticSMOService


def main() -> None:
    settings = get_settings()
    service = AgenticSMOService(settings)
    output_dir = settings.output_dir

    scores, summary_df = run_repeatability_experiment(service)
    save_repeatability_artifacts(scores, summary_df, output_dir)

    repeat_cost_df = build_run_level_cost_dataframe(service.prompt_logger.bucket("repeatability"))
    repeat_cost_df.to_csv(output_dir / "cost_repeatability.csv", index=False)
    save_prompt_log_json(
        service.prompt_logger.bucket("repeatability"),
        output_dir / "prompt_log_repeatability.json",
    )

    print(f"Saved repeatability outputs to {output_dir}")


if __name__ == "__main__":
    main()
