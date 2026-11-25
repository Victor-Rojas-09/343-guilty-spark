#!/usr/bin/env python3
"""Consulta de humo contra la colección vectorial de PostgreSQL/pgvector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spark_rag.config import get_settings
from spark_rag.db.connection import check_database, get_engine
from spark_rag.ingestion.embeddings import get_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca fragmentos similares en la colección pgvector."
    )
    parser.add_argument("query", help="Texto o concepto a buscar.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Número de fragmentos a devolver (por defecto 5).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filtra por categoría, p. ej. history, dialogs, sources o community.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    status = check_database()
    if not status.ok:
        print(
            "PostgreSQL/pgvector no está disponible. "
            f"Detalle: {status.error or 'desconocido'}",
            file=sys.stderr,
        )
        return 2

    from langchain_postgres import PGVector

    vector_store = PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.collection_name,
        connection=get_engine(),
        embedding_length=settings.embedding_dim,
        use_jsonb=True,
    )

    search_kwargs: dict[str, object] = {"k": max(1, args.top_k)}
    if args.category:
        search_kwargs["filter"] = {"category": args.category}

    results = vector_store.similarity_search_with_score(
        args.query,
        **search_kwargs,
    )
    if not results:
        print("No se encontraron fragmentos para la consulta.")
        return 1

    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata
        print(
            f"\n#{rank} similitud={1 - float(distance):.4f} "
            f"source={metadata.get('source')} "
            f"category={metadata.get('category')} "
            f"page={metadata.get('page')}"
        )
        snippet = " ".join(document.page_content.split())
        print(snippet[:500])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
