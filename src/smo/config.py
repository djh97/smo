from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def _split_paths(raw: str) -> list[Path]:
    return [Path(item.strip()).expanduser() for item in raw.split(";") if item.strip()]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    guideline_dir: Path = PROJECT_ROOT / "data" / "guidelines"
    index_dir: Path = PROJECT_ROOT / "data" / "indices" / "faiss_guidelines_index"
    saved_result_dir: Path = PROJECT_ROOT / "data" / "patient_records"
    output_dir: Path = PROJECT_ROOT / "outputs" / "evaluation"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    openai_model: str = os.getenv("SMO_OPENAI_MODEL", "gpt-4o")
    anthropic_model: str = os.getenv("SMO_ANTHROPIC_MODEL", "claude-opus-4-6")
    google_model: str = os.getenv("SMO_GOOGLE_MODEL", "gemini-2.5-flash")
    embedding_model: str | None = os.getenv("SMO_EMBEDDING_MODEL", "text-embedding-ada-002") or None
    top_k: int = int(os.getenv("SMO_TOP_K", "3"))
    chunk_size: int = int(os.getenv("SMO_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("SMO_CHUNK_OVERLAP", "100"))
    temperature: float = float(os.getenv("SMO_TEMPERATURE", "0.2"))
    geval_judge_model: str = os.getenv("SMO_G_EVAL_JUDGE_MODEL", "gpt-4o")
    geval_threshold: float = float(os.getenv("SMO_G_EVAL_THRESHOLD", "0.90"))
    geval_threshold_points: int = int(os.getenv("SMO_G_EVAL_THRESHOLD_POINTS", "201"))
    geval_threshold_rule: str = os.getenv(
        "SMO_G_EVAL_THRESHOLD_RULE",
        "Manual fixed threshold",
    )

    def guideline_paths(self) -> list[Path]:
        configured = os.getenv("SMO_GUIDELINE_PATHS", "").strip()
        if configured:
            return _split_paths(configured)

        expected = [
            self.guideline_dir / "guideline-170-en-61-113.pdf",
            self.guideline_dir / "TUV_D1_RESPIRATORY GUIDELINES.pdf",
        ]
        existing_expected = [path for path in expected if path.exists()]
        if existing_expected:
            return existing_expected

        return sorted(self.guideline_dir.glob("*.pdf"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
