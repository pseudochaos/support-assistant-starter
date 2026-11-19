"""Triage of one customer message: the module is yours to write (tasks 1 and 3)."""

from langchain_core.language_models import BaseChatModel


def handle(model: BaseChatModel, message: str, prompt_version: str) -> str:
    """Triage one customer message and return the reply to the customer.

    Expected flow: call the model with structured output against your schema,
    let code choose who handles the message, write one line to calls.jsonl,
    and turn any failure into the fallback to a human operator.
    """
    raise NotImplementedError("handle() is not implemented yet: tasks 1 and 3")
