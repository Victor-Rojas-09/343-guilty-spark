"""Esquema y operaciones de control de la ingesta.

La colección vectorial la gestiona `langchain-postgres` en sus propias tablas
(`langchain_pg_collection` y `langchain_pg_embedding`). Este módulo crea las
tablas de control que permiten implementar la idempotencia por hash SHA256 y
dejar un reporte persistente de cada ejecución.
"""

from __future__ import annotations

from typing import Any

import psycopg

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE EXTENSION IF NOT EXISTS vector
    """,
    """
    CREATE TABLE IF NOT EXISTS ingested_files (
        id           BIGSERIAL PRIMARY KEY,
        source       TEXT NOT NULL UNIQUE,
        sha256       CHAR(64),
        status       TEXT NOT NULL
                     CHECK (status IN ('indexed', 'failed')),
        file_size    BIGINT NOT NULL DEFAULT 0,
        chunk_count  INTEGER NOT NULL DEFAULT 0,
        error        TEXT,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS ingested_files_status_idx
        ON ingested_files (status)
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_runs (
        id               BIGSERIAL PRIMARY KEY,
        started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
        finished_at      TIMESTAMPTZ,
        status           TEXT NOT NULL DEFAULT 'running',
        files_discovered INTEGER NOT NULL DEFAULT 0,
        files_indexed    INTEGER NOT NULL DEFAULT 0,
        files_skipped    INTEGER NOT NULL DEFAULT 0,
        files_failed     INTEGER NOT NULL DEFAULT 0
    )
    """,
)


def initialize_schema(connection: psycopg.Connection) -> None:
    """Habilita pgvector y crea las tablas de control de forma idempotente."""
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)


def ensure_vector_index(connection: psycopg.Connection) -> bool:
    """Crea un índice HNSW para la colección vectorial si la tabla ya existe.

    `langchain-postgres` crea las tablas vectoriales de forma perezosa. Por
    eso este método se invoca después de instanciar el vector store.
    """
    row = connection.execute(
        "SELECT to_regclass('public.langchain_pg_embedding') AS table_name"
    ).fetchone()
    if row is None or row["table_name"] is None:
        return False

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS langchain_pg_embedding_hnsw_idx
            ON langchain_pg_embedding
            USING hnsw (embedding vector_cosine_ops)
        """
    )
    return True


def get_file_record(
    connection: psycopg.Connection, source: str
) -> dict[str, Any] | None:
    """Devuelve el registro de control de un archivo o `None` si no existe."""
    return connection.execute(
        """
        SELECT source, sha256, status, file_size, chunk_count, error, updated_at
        FROM ingested_files
        WHERE source = %s
        """,
        (source,),
    ).fetchone()


def upsert_file_record(
    connection: psycopg.Connection,
    *,
    source: str,
    sha256: str | None,
    status: str,
    file_size: int,
    chunk_count: int = 0,
    error: str | None = None,
) -> None:
    """Inserta o actualiza el estado de indexación de un archivo."""
    if status not in {"indexed", "failed"}:
        raise ValueError(f"Estado de archivo no válido: {status!r}")

    connection.execute(
        """
        INSERT INTO ingested_files (
            source, sha256, status, file_size, chunk_count, error
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (source) DO UPDATE SET
            sha256 = EXCLUDED.sha256,
            status = EXCLUDED.status,
            file_size = EXCLUDED.file_size,
            chunk_count = EXCLUDED.chunk_count,
            error = EXCLUDED.error,
            updated_at = now()
        """,
        (source, sha256, status, file_size, chunk_count, error),
    )


def delete_vectors_for_source(
    connection: psycopg.Connection, *, collection_name: str, source: str
) -> int:
    """Elimina de la colección vectorial todos los chunks de un `source`.

    Se usa SQL directo porque `PGVector.delete()` de LangChain solo permite
    borrar por identificadores, no por filtro de metadatos.
    """
    cursor = connection.execute(
        """
        DELETE FROM langchain_pg_embedding AS embedding
        USING langchain_pg_collection AS collection
        WHERE embedding.collection_id = collection.uuid
          AND collection.name = %s
          AND embedding.cmetadata->>'source' = %s
        """,
        (collection_name, source),
    )
    return cursor.rowcount or 0


def create_run(connection: psycopg.Connection, files_discovered: int) -> int:
    """Registra el inicio de una ejecución de ingesta y devuelve su id."""
    row = connection.execute(
        """
        INSERT INTO ingestion_runs (files_discovered, status)
        VALUES (%s, 'running')
        RETURNING id
        """,
        (files_discovered,),
    ).fetchone()
    if row is None:  # pragma: no cover - defensivo
        raise RuntimeError("No se pudo registrar la ejecución de ingesta.")
    return int(row["id"])


def finish_run(
    connection: psycopg.Connection,
    *,
    run_id: int,
    status: str,
    files_discovered: int,
    files_indexed: int,
    files_skipped: int,
    files_failed: int,
) -> None:
    """Cierra una ejecución de ingesta con sus contadores finales."""
    connection.execute(
        """
        UPDATE ingestion_runs
        SET finished_at = now(),
            status = %s,
            files_discovered = %s,
            files_indexed = %s,
            files_skipped = %s,
            files_failed = %s
        WHERE id = %s
        """,
        (
            status,
            files_discovered,
            files_indexed,
            files_skipped,
            files_failed,
            run_id,
        ),
    )
