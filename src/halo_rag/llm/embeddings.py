"""Generación de embeddings con Gemini.

Responsabilidad:
    Convertir textos en vectores usando el modelo configurado, procesando en
    lotes y respetando los límites de la capa gratuita.

Nota: la implementación concreta corresponde a la Fase 2.
"""


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para un lote de textos.

    Raises:
        NotImplementedError: Pendiente de implementación en la Fase 2.
    """
    raise NotImplementedError("Embeddings pendientes de implementación (Fase 2).")


def embed_query(text: str) -> list[float]:
    """Genera el embedding de una consulta del usuario.

    Raises:
        NotImplementedError: Pendiente de implementación en la Fase 2.
    """
    raise NotImplementedError("Embedding de consulta pendiente (Fase 2).")
