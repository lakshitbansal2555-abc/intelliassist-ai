"""
vector_store.py
================
"Semantic Search using Embeddings" + the retrieval half of RAG.

Uses a local sentence-transformers model (all-MiniLM-L6-v2, a distilled
BERT model - satisfies the "BERT Embeddings" line in the tech stack) so
embedding documents never needs an API key or costs money. FAISS stores
the vectors and does the nearest-neighbour search.
"""

from __future__ import annotations

import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "vector_store"


_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily loads the embedding model once per process (it's ~90MB)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def build_vector_store(chunks: list[Document]) -> FAISS:
    """Embeds every chunk and builds a fresh in-memory FAISS index."""
    return FAISS.from_documents(chunks, get_embeddings())


def save_vector_store(store: FAISS, name: str = "index") -> None:
    """Optional persistence, so a processed session survives an app restart."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    store.save_local(INDEX_DIR, index_name=name)


def load_vector_store(name: str = "index") -> FAISS | None:
    path = os.path.join(INDEX_DIR, f"{name}.faiss")
    if not os.path.exists(path):
        return None
    return FAISS.load_local(
        INDEX_DIR, get_embeddings(), index_name=name, allow_dangerous_deserialization=True
    )


def semantic_search(store: FAISS, query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Returns (chunk, similarity_score) pairs, most relevant first."""
    return store.similarity_search_with_relevance_scores(query, k=k)
