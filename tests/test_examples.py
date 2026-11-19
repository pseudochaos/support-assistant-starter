"""Examples of tests for your own code. This file is yours: change it with your code."""

import pytest

from support_assistant.config import load_model
from support_assistant.prompts import load_prompt


def test_prompt_is_read_from_versioned_file(tmp_path):
    (tmp_path / "v1.md").write_text("Ти — асистент техпідтримки.\n", encoding="utf-8")

    assert load_prompt("v1", tmp_path) == "Ти — асистент техпідтримки."
    with pytest.raises(FileNotFoundError):
        load_prompt("v2", tmp_path)


def test_model_comes_from_configuration(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    model = load_model("google_genai:gemini-3.5-flash-lite")

    assert "gemini-3.5-flash-lite" in model.model
