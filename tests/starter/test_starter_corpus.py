"""The corpus is in place. Course file: do not edit, the next practical may update it."""

from support_assistant.kb import KB_DIR


def test_corpus_is_in_place():
    assert len(list(KB_DIR.glob("*.md"))) == 40
