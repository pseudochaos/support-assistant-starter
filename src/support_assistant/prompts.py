"""System instructions are versioned files in prompts/, not strings in code."""

from pathlib import Path

from .config import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "prompts"


def load_prompt(version: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    """Text of prompts/<version>.md; the version also goes into every log line."""
    path = prompts_dir / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"System instructions {path.name} not found in {prompts_dir}")
    return path.read_text(encoding="utf-8").strip()
