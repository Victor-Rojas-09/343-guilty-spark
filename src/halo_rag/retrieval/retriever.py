"""Búsqueda de fragmentos relevantes en pgvector.

Responsabilidad:
    Dado el embedding de una consulta, recuperar los `top_k` chunks más
    similares usando el operador de distancia elegido, con documento y página.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """Fragmento recuperado de la base de datos vectorial."""

    content: str
    page_number: int
    document_title: str
    distance: float


def search(query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
    """Recupera los chunks más similares a un embedding de consulta.

    Raises:
        NotImplementedError: Pendiente de implementación en la Fase 2.
    """
    raise NotImplementedError("Búsqueda vectorial pendiente de implementación (Fase 2).")
