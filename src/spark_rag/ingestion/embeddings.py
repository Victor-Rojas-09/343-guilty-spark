"""Embeddings de Gemini con validación de dimensión y valores finitos."""

from __future__ import annotations

import math
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from spark_rag.config import get_settings


def validate_vectors(
    vectors: list[list[float]],
    *,
    expected_dim: int,
    label: str = "embedding",
) -> None:
    """Valida que los vectores no estén vacíos, sean finitos y tengan dimensión.

    Raises:
        ValueError: Si algún vector incumple las condiciones.
    """
    if not vectors:
        raise ValueError(f"No se generó ningún {label}.")

    for index, vector in enumerate(vectors):
        if not vector:
            raise ValueError(f"El {label} {index} está vacío.")
        if len(vector) != expected_dim:
            raise ValueError(
                f"El {label} {index} tiene dimensión {len(vector)}, "
                f"se esperaba {expected_dim}."
            )
        for value in vector:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"El {label} {index} contiene un valor no finito.")


class ValidatedEmbeddings(Embeddings):
    """Envuelve un modelo de LangChain y valida sus vectores.

    La dimensión solicitada se pasa en cada llamada porque las versiones de
    `langchain-google-genai` compatibles con esta pila la exponen en
    `embed_documents`/`embed_query`, no siempre en el constructor.
    """

    def __init__(self, inner: Embeddings, expected_dim: int) -> None:
        self._inner = inner
        self._expected_dim = expected_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._inner.embed_documents(
            texts,
            output_dimensionality=self._expected_dim,
        )
        validate_vectors(
            vectors,
            expected_dim=self._expected_dim,
            label="embedding de documento",
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._inner.embed_query(
            text,
            output_dimensionality=self._expected_dim,
        )
        validate_vectors(
            [vector],
            expected_dim=self._expected_dim,
            label="embedding de consulta",
        )
        return vector


def build_embeddings() -> Embeddings:
    """Construye el modelo de embeddings de Gemini configurado."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY no está configurada. Añádela a .env "
            "o al entorno del proceso."
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = settings.gemini_embedding_model
    if not model.startswith("models/"):
        model = f"models/{model}"

    inner = GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=settings.gemini_api_key,
    )
    return ValidatedEmbeddings(inner, settings.embedding_dim)


@lru_cache
def get_embeddings() -> Embeddings:
    """Devuelve una instancia reutilizable del modelo de embeddings."""
    return build_embeddings()
