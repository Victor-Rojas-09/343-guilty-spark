"""Recuperación semántica sobre la colección pgvector.

Reutiliza el mismo modelo y dimensión de embeddings que la ingesta
(`ingestion.embeddings.get_embeddings`) y la misma colección de
`langchain-postgres`. La métrica es coseno, coherente con el índice HNSW
(`vector_cosine_ops`): `similarity_search_with_score` devuelve una distancia
`d` y aquí se expone la similitud `1 - d`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

from langchain_core.runnables import Runnable, RunnableLambda

from spark_rag.config import Settings, get_settings
from spark_rag.rag.sources import RetrievedChunk

logger = logging.getLogger(__name__)


class SupportsScoredSearch(Protocol):
    """Subconjunto de la interfaz de `PGVector` que usa el recuperador."""

    def similarity_search_with_score(
        self, query: str, k: int = ..., filter: dict[str, Any] | None = ...
    ) -> list[tuple[Any, float]]: ...


def _normalize_values(values: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def build_filter(
    *,
    categories: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Traduce filtros de categoría/fuente al formato JSONB de `PGVector`."""
    clauses: list[dict[str, Any]] = []
    category_values = [value.lower() for value in _normalize_values(categories)]
    source_values = _normalize_values(sources)
    if category_values:
        clauses.append({"category": {"$in": category_values}})
    if source_values:
        clauses.append({"source": {"$in": source_values}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def distance_to_similarity(distance: float) -> float:
    """Convierte distancia coseno (0..2) en similitud (-1..1)."""
    return max(-1.0, min(1.0, 1.0 - float(distance)))


class Retriever:
    """Recupera fragmentos pertinentes aplicando top-k, umbral y filtros."""

    def __init__(
        self,
        vector_store: SupportsScoredSearch,
        *,
        top_k: int,
        min_similarity: float,
        categories: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k debe ser al menos 1.")
        self._vector_store = vector_store
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.categories = _normalize_values(categories)
        self.sources = _normalize_values(sources)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        categories: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Devuelve los fragmentos con similitud >= umbral, de mayor a menor.

        Los filtros pasados como argumento sustituyen a los configurados.
        """
        text = query.strip()
        if not text:
            return []

        search_filter = build_filter(
            categories=categories if categories is not None else self.categories,
            sources=sources if sources is not None else self.sources,
        )
        kwargs: dict[str, Any] = {"k": top_k or self.top_k}
        if search_filter is not None:
            kwargs["filter"] = search_filter

        results = self._vector_store.similarity_search_with_score(text, **kwargs)

        chunks: list[RetrievedChunk] = []
        for document, distance in results:
            similarity = distance_to_similarity(distance)
            if similarity < self.min_similarity:
                continue
            chunks.append(
                RetrievedChunk(
                    content=getattr(document, "page_content", "") or "",
                    similarity=similarity,
                    metadata=dict(getattr(document, "metadata", {}) or {}),
                )
            )
        chunks.sort(key=lambda chunk: chunk.similarity, reverse=True)

        logger.info(
            "Recuperación: %d candidatos, %d sobre el umbral %.2f",
            len(results),
            len(chunks),
            self.min_similarity,
        )
        return chunks

    def as_runnable(self) -> Runnable[str, list[RetrievedChunk]]:
        """Expone la recuperación como paso de una cadena LangChain."""
        return RunnableLambda(self.retrieve, name="spark_retriever")


@lru_cache
def get_vector_store() -> Any:
    """Instancia reutilizable de `PGVector` sobre la colección configurada."""
    from langchain_postgres import PGVector

    from spark_rag.db.connection import get_engine
    from spark_rag.ingestion.embeddings import get_embeddings

    settings = get_settings()
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.collection_name,
        connection=get_engine(),
        embedding_length=settings.embedding_dim,
        use_jsonb=True,
    )


def build_retriever(
    settings: Settings | None = None,
    *,
    vector_store: SupportsScoredSearch | None = None,
) -> Retriever:
    """Construye el recuperador a partir de la configuración."""
    settings = settings or get_settings()
    return Retriever(
        vector_store if vector_store is not None else get_vector_store(),
        top_k=settings.retrieval_top_k,
        min_similarity=settings.retrieval_min_similarity,
        categories=settings.retrieval_category_list,
    )
