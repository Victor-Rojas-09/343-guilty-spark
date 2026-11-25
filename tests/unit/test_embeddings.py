import math

import pytest

from spark_rag.ingestion.embeddings import ValidatedEmbeddings, validate_vectors


def test_acepta_vectores_validos() -> None:
    validate_vectors([[0.1, 0.2, 0.3]], expected_dim=3)


def test_rechaza_dimension_incorrecta() -> None:
    with pytest.raises(ValueError, match="dimensión"):
        validate_vectors([[0.1, 0.2]], expected_dim=3)


def test_rechaza_vectores_vacios() -> None:
    with pytest.raises(ValueError, match="vacío"):
        validate_vectors([[]], expected_dim=3)
    with pytest.raises(ValueError, match="ningún"):
        validate_vectors([], expected_dim=3)


def test_rechaza_valores_no_finitos() -> None:
    with pytest.raises(ValueError, match="no finito"):
        validate_vectors([[0.1, math.nan, 0.3]], expected_dim=3)


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.doc_kwargs: dict[str, object] = {}
        self.query_kwargs: dict[str, object] = {}

    def embed_documents(self, texts, **kwargs):
        self.doc_kwargs = kwargs
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text, **kwargs):
        self.query_kwargs = kwargs
        return [0.1, 0.2, 0.3]


def test_wrapper_solicita_la_dimension_configurada() -> None:
    inner = _FakeEmbeddings()
    embeddings = ValidatedEmbeddings(inner, expected_dim=3)

    assert embeddings.embed_documents(["uno"]) == [[0.1, 0.2, 0.3]]
    assert embeddings.embed_query("dos") == [0.1, 0.2, 0.3]
    assert inner.doc_kwargs["output_dimensionality"] == 3
    assert inner.query_kwargs["output_dimensionality"] == 3
