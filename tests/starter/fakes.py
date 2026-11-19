"""Stand-ins for the provider: tests run without network or API key.

Course file: do not edit, the next practical may update it.
"""

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


class InvalidOutput:
    """A model reply that does not match the schema."""

    def __init__(self, text: str):
        self.text = text


class FakeModel(BaseChatModel):
    """Returns scripted replies in order and records what it received.

    A reply is one of: an instance of your schema (valid structured output),
    InvalidOutput (text that fails validation), an exception (a provider error
    or a timeout, raised as is) or a string (a plain text answer).
    """

    replies: list = Field(default_factory=list)
    received: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _next(self, prompt):
        self.received.append(prompt)
        if not self.replies:
            raise AssertionError("FakeModel has no scripted replies left")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self._next(messages)
        text = reply.text if isinstance(reply, InvalidOutput) else str(reply)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text, usage_metadata=USAGE))])

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        """Mirrors the real contract, including include_raw=True with raw, parsed and parsing_error."""

        def call(prompt):
            reply = self._next(prompt)
            if isinstance(reply, InvalidOutput):
                raw = AIMessage(reply.text, usage_metadata=USAGE)
                error = OutputParserException(f"Invalid structured output: {reply.text!r}", llm_output=reply.text)
                if include_raw:
                    return {"raw": raw, "parsed": None, "parsing_error": error}
                raise error
            text = reply.model_dump_json() if isinstance(reply, BaseModel) else str(reply)
            raw = AIMessage(text, usage_metadata=USAGE)
            return {"raw": raw, "parsed": reply, "parsing_error": None} if include_raw else reply

        return RunnableLambda(call)


class FakeEmbeddings(DeterministicFakeEmbedding):
    """DeterministicFakeEmbedding from LangChain that also records every text it embeds.

    The same text always gets the same vector, and different texts get unrelated
    ones: similarity here says nothing about meaning. To make a fragment the
    nearest one to a question in a test, ask a question equal to its text.
    `embedded` lists every text sent for embedding, in order: a re-index of an
    unchanged corpus through your cache must not add anything to it.
    The vector also depends on `model`, so two fakes with different `model`
    are incompatible, like two real embedding models.
    """

    model: str = "fake-embedding"
    size: int = 64
    embedded: list = Field(default_factory=list)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return super().embed_documents([f"{self.model}:{t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        self.embedded.append(text)
        return super().embed_query(f"{self.model}:{text}")
