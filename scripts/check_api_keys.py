from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from smo.config import get_settings


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"{name}: OK")
        return True
    except Exception as exc:
        print(f"{name}: FAIL -> {exc.__class__.__name__}: {exc}")
        return False


def main() -> int:
    settings = get_settings()
    success = True

    if settings.openai_api_key:
        success &= check(
            "OpenAI chat",
            lambda: ChatOpenAI(
                model=settings.openai_model,
                temperature=0,
                max_tokens=1,
                api_key=settings.openai_api_key,
            ).invoke("Reply with OK"),
        )
        success &= check(
            "OpenAI embeddings",
            lambda: OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            ).embed_query("ping"),
        )
    else:
        print("OpenAI chat: SKIP -> OPENAI_API_KEY is missing.")
        print("OpenAI embeddings: SKIP -> OPENAI_API_KEY is missing.")
        success = False

    if settings.anthropic_api_key:
        success &= check(
            "Anthropic",
            lambda: ChatAnthropic(
                model=settings.anthropic_model,
                temperature=0,
                max_tokens=1,
                api_key=settings.anthropic_api_key,
            ).invoke("Reply with OK"),
        )
    else:
        print("Anthropic: SKIP -> ANTHROPIC_API_KEY is missing.")

    if settings.google_api_key:
        success &= check(
            "Google",
            lambda: ChatGoogleGenerativeAI(
                model=settings.google_model,
                temperature=0,
                max_output_tokens=1,
                google_api_key=settings.google_api_key,
            ).invoke("Reply with OK"),
        )
    else:
        print("Google: SKIP -> GOOGLE_API_KEY is missing.")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
