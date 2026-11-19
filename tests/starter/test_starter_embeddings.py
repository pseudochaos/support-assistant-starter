"""How the fake embedding model behaves. Course file: do not edit, the next practical may update it."""

import pytest
from fakes import FakeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


def test_fake_embeddings_give_the_same_vector_to_the_same_text():
    embeddings = FakeEmbeddings()

    assert embeddings.embed_query("Як скинути пароль?") == embeddings.embed_documents(["Як скинути пароль?"])[0]
    assert embeddings.embed_query("Як скинути пароль?") != embeddings.embed_query("Як змінити тариф?")


def test_fake_embeddings_make_a_fragment_nearest_by_its_own_text():
    store = InMemoryVectorStore(FakeEmbeddings())
    store.add_documents([
        Document("Посилання дійсне 30 хвилин.", id="password-reset#how-to-reset"),
        Document("Перехід на дешевший тариф діє з наступного періоду.", id="plans#change-plan"),
    ])

    (doc, score), _ = store.similarity_search_with_score("Посилання дійсне 30 хвилин.", k=2)

    assert doc.id == "password-reset#how-to-reset"
    assert score == pytest.approx(1.0)


def test_fake_embeddings_record_every_embedded_text():
    embeddings = FakeEmbeddings()

    InMemoryVectorStore(embeddings).add_documents([Document("перший"), Document("другий")])

    assert embeddings.embedded == ["перший", "другий"]


def test_fake_embeddings_of_different_models_are_incompatible():
    assert FakeEmbeddings(model="a").embed_query("текст") != FakeEmbeddings(model="b").embed_query("текст")
