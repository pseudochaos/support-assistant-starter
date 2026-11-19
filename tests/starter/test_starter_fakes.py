"""How the fake chat model behaves. Course file: do not edit, the next practical may update it."""

from typing import Literal

import pytest
from fakes import USAGE, FakeModel, InvalidOutput
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel


class Example(BaseModel):
    category: Literal["A", "B"]


def test_fake_returns_scripted_structured_output():
    model = FakeModel(replies=[Example(category="A")])

    assert model.with_structured_output(Example).invoke("Як скинути пароль?") == Example(category="A")
    assert model.received == ["Як скинути пароль?"]


def test_fake_invalid_output_raises_without_include_raw():
    model = FakeModel(replies=[InvalidOutput('{"category": "Z"}')])

    with pytest.raises(OutputParserException):
        model.with_structured_output(Example).invoke("...")


def test_fake_invalid_output_with_include_raw_keeps_tokens():
    model = FakeModel(replies=[InvalidOutput("not json")])

    result = model.with_structured_output(Example, include_raw=True).invoke("...")

    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)
    assert result["raw"].usage_metadata == USAGE


def test_fake_raises_provider_errors_as_is():
    model = FakeModel(replies=[TimeoutError("timed out")])

    with pytest.raises(TimeoutError):
        model.with_structured_output(Example).invoke("...")
