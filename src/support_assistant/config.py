"""Settings from .env: provider, model and prompt version live in configuration, not in code."""

import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "google_genai:gemini-3.5-flash-lite"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_settings() -> None:
    load_dotenv(REPO_ROOT / ".env")


def model_name() -> str:
    return os.getenv("ASSISTANT_MODEL", DEFAULT_MODEL)


def prompt_version() -> str:
    return os.getenv("PROMPT_VERSION", "v1")


def load_model(name: str):
    from langchain.chat_models import init_chat_model

    # TODO (task 1): give every call a timeout and a bounded number of retries,
    # both read from .env. Without a timeout a stalled call never reaches the fallback.
    return init_chat_model(name)
