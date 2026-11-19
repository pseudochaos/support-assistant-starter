"""Embedding model from .env: like the chat model, it lives in configuration, not in code.

Embeddings of different models are incompatible: after changing EMBEDDING_MODEL
the whole corpus has to be embedded again, never mixed with old vectors.
"""

import os

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


def embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def load_embeddings(name: str):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # gemini-embedding-001 gets the task type (document or query) from LangChain itself.
    # TODO (task 1): every call goes to the API and spends the free-tier quota;
    # put a disk cache keyed by model + hash of the fragment text in front of it.
    return GoogleGenerativeAIEmbeddings(model=name)
