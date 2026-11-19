"""Contract of KnowledgeBase that the next practicals rely on: its shape, not its behaviour.

Skipped while KnowledgeBase raises NotImplementedError. Once it is implemented,
these tests check only what search() returns; access rules, re-indexing and
citations are for your own tests (task 5). Course file: do not edit.
"""

import pytest
from fakes import FakeEmbeddings
from langchain_core.documents import Document

from support_assistant.kb import KnowledgeBase

ARTICLES = {
    "first.md": """---
id: alpha
title: Перша стаття
updated: 2026-09-01
audience: customer
owner: support
---
# Перша стаття

## Розділ один {#one}

Як увімкнути нагадування про строки подання звітів.

## Розділ два {#two}

Як вимкнути нагадування електронною поштою.
""",
    "second.md": """---
id: beta
title: Друга стаття
updated: 2026-09-01
audience: customer
owner: support
---
# Друга стаття

## Розділ один {#one}

Нагадування надходять за сім днів до строку.
""",
}
METHODS = ["bm25", "semantic", "hybrid"]


@pytest.fixture
def kb(tmp_path):
    corpus = tmp_path / "kb"
    corpus.mkdir()
    for name, text in ARTICLES.items():
        (corpus / name).write_text(text, encoding="utf-8")
    try:
        return KnowledgeBase(corpus, FakeEmbeddings(), tmp_path / "cache")
    except NotImplementedError as e:
        pytest.skip(str(e))


def search(kb, question, **kwargs):
    try:
        return kb.search(question, **kwargs)
    except NotImplementedError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("method", METHODS)
def test_search_returns_at_most_k_documents(kb, method):
    found = search(kb, "Як увімкнути нагадування?", audience="customer", method=method, k=2)

    assert isinstance(found, list)
    assert 0 < len(found) <= 2
    assert all(isinstance(doc, Document) for doc in found)


@pytest.mark.parametrize("method", METHODS)
def test_fragments_carry_stable_ids_and_article_metadata(kb, method):
    found = search(kb, "Як увімкнути нагадування?", audience="customer", method=method, k=3)

    ids = [doc.id for doc in found]
    assert all(ids) and len(set(ids)) == len(ids)
    for doc in found:
        assert doc.metadata["article_id"] in {"alpha", "beta"}
        assert doc.metadata["article_id"] in doc.id
        assert doc.metadata["title"] in {"Перша стаття", "Друга стаття"}
        assert doc.metadata["audience"] == "customer"
        assert doc.page_content


def test_k_defaults_to_configuration(kb):
    assert isinstance(search(kb, "Як увімкнути нагадування?", audience="customer"), list)
