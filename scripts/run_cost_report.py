from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smo.config import get_settings
from smo.evaluation import (
    build_dynamic_cost_breakdown_dataframe,
    build_dynamic_latency_breakdown_dataframe,
    run_alignment_experiment,
    save_cost_artifacts,
    save_latency_artifacts,
    save_prompt_log_json,
)
from smo.pipeline import AgenticSMOService


def main() -> None:
    settings = get_settings()
    service = AgenticSMOService(settings)
    output_dir = settings.output_dir

    run_alignment_experiment(service)
    cost_df = build_dynamic_cost_breakdown_dataframe(service.prompt_logger.bucket("four_case"))
    latency_df = build_dynamic_latency_breakdown_dataframe(service.prompt_logger.bucket("four_case"))
    save_cost_artifacts(cost_df, output_dir)
    save_latency_artifacts(latency_df, output_dir)
    save_prompt_log_json(service.prompt_logger.bucket("four_case"), output_dir / "prompt_log_4case.json")

    print(f"Saved cost outputs to {output_dir}")


if __name__ == "__main__":
    main()
