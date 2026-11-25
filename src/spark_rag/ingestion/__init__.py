"""Pipeline de ingesta documental: carga, limpieza, metadatos y embeddings."""

from spark_rag.ingestion.cleaner import clean_text
from spark_rag.ingestion.metadata import (
    build_document_metadata,
    compute_sha256,
    make_chunk_id,
)

__all__ = [
    "build_document_metadata",
    "clean_text",
    "compute_sha256",
    "make_chunk_id",
]
