"""Knowledge base: indexing and retrieval are yours to write (tasks 1 and 2).

The corpus is kb/*.md, one article per file with metadata in the frontmatter;
docs/knowledge-base.md describes the fields. Not every article follows the rules.

Contract: the next practicals and the course tests in tests/starter/ create
KnowledgeBase(kb_dir, embeddings, cache_dir) and call search() exactly as
declared below. Keep both signatures and what search() returns; everything
behind them is your design.
"""

from datetime import date
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import REPO_ROOT

KB_DIR = REPO_ROOT / "kb"

Audience = Literal["customer", "operator"]
Method = Literal["bm25", "semantic", "hybrid"]


class KnowledgeBase:
    """Index of the corpus with three retrieval methods behind one interface."""

    def __init__(self, kb_dir: Path, embeddings: Embeddings, cache_dir: Path) -> None:
        """Load and check the articles in kb_dir, split them and index the fragments.

        Embeddings go through a disk cache in cache_dir: building the index again
        over an unchanged corpus makes no calls to the embedding model.
        """
        raise NotImplementedError("KnowledgeBase is not implemented yet: task 1")

    def search(
        self,
        question: str,
        *,
        audience: Audience,
        method: Method = "hybrid",
        k: int | None = None,
        period: date | None = None,
    ) -> list[Document]:
        """Up to k fragments for the question, best first, that this audience may see.

        Each fragment is a Document: a stable id built from the article id, the
        fragment text in page_content, and in metadata at least "article_id",
        "title" and "audience". k defaults to the configured value. period is
        for variant 3: the reporting period the question is about.
        """
        raise NotImplementedError("KnowledgeBase.search() is not implemented yet: task 2")
