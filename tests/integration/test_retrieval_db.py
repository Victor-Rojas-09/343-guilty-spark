"""Recuperación contra pgvector real con embeddings falsos (sin Gemini)."""

import hashlib
import math
import os
import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from spark_rag.config import get_settings
from spark_rag.db.connection import get_engine
from spark_rag.rag.retriever import Retriever

pytestmark = pytest.mark.integration

requires_database = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="Define RUN_INTEGRATION=1 con PostgreSQL/pgvector disponible.",
)


class _BagOfWordsEmbeddings(Embeddings):
    """Embeddings deterministas: bolsa de palabras con hashing, normalizada."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def vector_store():
    from langchain_postgres import PGVector

    settings = get_settings()
    store = PGVector(
        embeddings=_BagOfWordsEmbeddings(settings.embedding_dim),
        collection_name=f"test_retrieval_{uuid.uuid4().hex[:8]}",
        connection=get_engine(),
        embedding_length=settings.embedding_dim,
        use_jsonb=True,
    )
    store.add_documents(
        [
            Document(
                page_content="chakas erde tyrene patron monitor",
                metadata={"source": "history/a.pdf", "category": "history", "page": 1},
            ),
            Document(
                page_content="indice instalacion activacion halo protocolo",
                metadata={"source": "dialogs/b.pdf", "category": "dialogs", "page": 4},
            ),
            Document(
                page_content="chakas teoria reddit debate identidad",
                metadata={
                    "source": "community/c.pdf",
                    "category": "community",
                    "page": 2,
                },
            ),
        ]
    )
    yield store
    store.delete_collection()


@requires_database
def test_recupera_fragmento_pertinente_con_metadatos(vector_store) -> None:
    retriever = Retriever(vector_store, top_k=3, min_similarity=0.3)
    chunks = retriever.retrieve("chakas patron monitor")

    assert chunks
    assert chunks[0].source == "history/a.pdf"
    assert chunks[0].page == 1
    assert all(chunk.similarity >= 0.3 for chunk in chunks)


@requires_database
def test_umbral_descarta_consultas_sin_evidencia(vector_store) -> None:
    retriever = Retriever(vector_store, top_k=3, min_similarity=0.3)
    assert retriever.retrieve("cortana mjolnir spartan") == []


@requires_database
def test_filtro_por_categoria(vector_store) -> None:
    retriever = Retriever(
        vector_store, top_k=3, min_similarity=0.0, categories=["community"]
    )
    chunks = retriever.retrieve("chakas")

    assert chunks
    assert {chunk.category for chunk in chunks} == {"community"}


@requires_database
def test_filtro_por_fuente(vector_store) -> None:
    retriever = Retriever(vector_store, top_k=3, min_similarity=0.0)
    chunks = retriever.retrieve("chakas", sources=["history/a.pdf"])

    assert [chunk.source for chunk in chunks] == ["history/a.pdf"]
