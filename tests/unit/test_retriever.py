from typing import Any

import pytest
from langchain_core.documents import Document

from spark_rag.config import Settings
from spark_rag.rag.retriever import (
    Retriever,
    build_filter,
    build_retriever,
    distance_to_similarity,
)


class _FakeVectorStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def similarity_search_with_score(self, query: str, **kwargs: Any):
        self.calls.append({"query": query, **kwargs})
        return self.results[: kwargs.get("k", len(self.results))]


def _doc(source: str, category: str = "history", page: int = 1) -> Document:
    return Document(
        page_content=f"contenido {source}",
        metadata={"source": source, "category": category, "page": page},
    )


def test_construye_filtros_jsonb() -> None:
    assert build_filter() is None
    assert build_filter(categories=["History", " "]) == {
        "category": {"$in": ["history"]}
    }
    assert build_filter(sources=["a.pdf", "a.pdf"]) == {"source": {"$in": ["a.pdf"]}}
    assert build_filter(categories=["dialogs"], sources=["b.pdf"]) == {
        "$and": [
            {"category": {"$in": ["dialogs"]}},
            {"source": {"$in": ["b.pdf"]}},
        ]
    }


@pytest.mark.parametrize(
    ("distance", "similarity"),
    [(0.0, 1.0), (0.25, 0.75), (1.0, 0.0), (2.5, -1.0)],
)
def test_distancia_a_similitud(distance: float, similarity: float) -> None:
    assert distance_to_similarity(distance) == pytest.approx(similarity)


def test_descarta_fragmentos_bajo_el_umbral_y_ordena() -> None:
    store = _FakeVectorStore(
        [
            (_doc("b.pdf"), 0.30),  # similitud 0.70
            (_doc("a.pdf"), 0.10),  # similitud 0.90
            (_doc("c.pdf"), 0.60),  # similitud 0.40 -> fuera
        ]
    )
    retriever = Retriever(store, top_k=5, min_similarity=0.5)
    chunks = retriever.retrieve("Chakas")

    assert [c.source for c in chunks] == ["a.pdf", "b.pdf"]
    assert chunks[0].similarity == pytest.approx(0.9)
    assert chunks[0].metadata["category"] == "history"


def test_aplica_top_k_y_filtros_configurados() -> None:
    store = _FakeVectorStore([(_doc("a.pdf"), 0.1)])
    retriever = Retriever(store, top_k=3, min_similarity=0.0, categories=["dialogs"])
    retriever.retrieve("  Índice  ")

    call = store.calls[0]
    assert call["query"] == "Índice"
    assert call["k"] == 3
    assert call["filter"] == {"category": {"$in": ["dialogs"]}}


def test_filtros_por_llamada_sustituyen_a_los_configurados() -> None:
    store = _FakeVectorStore([])
    retriever = Retriever(store, top_k=3, min_similarity=0.0, categories=["dialogs"])
    retriever.retrieve("x", top_k=7, categories=[], sources=["history/a.pdf"])

    call = store.calls[0]
    assert call["k"] == 7
    assert call["filter"] == {"source": {"$in": ["history/a.pdf"]}}


def test_sin_filtros_no_envia_filter() -> None:
    store = _FakeVectorStore([])
    Retriever(store, top_k=2, min_similarity=0.0).retrieve("x")
    assert "filter" not in store.calls[0]


def test_consulta_vacia_no_consulta_la_base() -> None:
    store = _FakeVectorStore([(_doc("a.pdf"), 0.1)])
    assert Retriever(store, top_k=2, min_similarity=0.0).retrieve("   ") == []
    assert store.calls == []


def test_top_k_invalido() -> None:
    with pytest.raises(ValueError):
        Retriever(_FakeVectorStore([]), top_k=0, min_similarity=0.5)


def test_runnable_de_langchain() -> None:
    store = _FakeVectorStore([(_doc("a.pdf"), 0.2)])
    runnable = Retriever(store, top_k=1, min_similarity=0.5).as_runnable()
    assert [c.source for c in runnable.invoke("Spark")] == ["a.pdf"]


def test_build_retriever_usa_la_configuracion() -> None:
    settings = Settings(
        _env_file=None,
        retrieval_top_k=4,
        retrieval_min_similarity=0.6,
        retrieval_categories="history,community",
    )
    retriever = build_retriever(settings, vector_store=_FakeVectorStore([]))

    assert retriever.top_k == 4
    assert retriever.min_similarity == 0.6
    assert retriever.categories == ["history", "community"]
