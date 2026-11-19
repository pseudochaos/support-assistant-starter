"""Examples of tests for your knowledge-base code. This file is yours: change it with your code."""

from support_assistant.embeddings import embedding_model_name, load_embeddings


def test_embedding_model_comes_from_configuration(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-001")

    assert load_embeddings(embedding_model_name()).model == "gemini-embedding-001"
