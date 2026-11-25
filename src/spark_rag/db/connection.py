"""Conexiones a PostgreSQL y comprobación de disponibilidad.

Se ofrecen dos accesos:

* `get_connection()`: conexión psycopg3 con filas tipo diccionario, usada para
  el control de ingesta y las comprobaciones de salud.
* `get_engine()`: motor SQLAlchemy reutilizable que consume `langchain-postgres`
  para la colección vectorial.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import psycopg
from psycopg.rows import dict_row
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from spark_rag.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Abre una conexión psycopg y la cierra al salir del contexto.

    El `with` de psycopg confirma la transacción si no hubo excepciones y hace
    rollback en caso contrario.
    """
    settings = get_settings()
    with psycopg.connect(settings.database_url, row_factory=dict_row) as connection:
        yield connection


@lru_cache
def get_engine() -> Engine:
    """Devuelve un motor SQLAlchemy reutilizable para LangChain."""
    settings = get_settings()
    return create_engine(
        settings.sqlalchemy_database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )


@dataclass(frozen=True)
class DatabaseStatus:
    """Resultado de la comprobación de base de datos y pgvector."""

    ok: bool
    server_version: str | None = None
    pgvector_version: str | None = None
    error: str | None = None


def check_database() -> DatabaseStatus:
    """Comprueba conexión, versión del servidor y extensión `vector`.

    No propaga excepciones para que `/health` pueda devolver un estado
    controlado. Los detalles técnicos se registran en el log del backend.
    """
    try:
        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    current_setting('server_version') AS server_version,
                    (
                        SELECT extversion
                        FROM pg_extension
                        WHERE extname = 'vector'
                    ) AS pgvector_version
                """
            ).fetchone()
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - entorno
        logger.warning("No se pudo comprobar PostgreSQL: %s", exc)
        return DatabaseStatus(ok=False, error=type(exc).__name__)

    if row is None:
        return DatabaseStatus(ok=False, error="respuesta_vacia")

    server_version = row["server_version"]
    pgvector_version = row["pgvector_version"]
    return DatabaseStatus(
        ok=pgvector_version is not None,
        server_version=server_version,
        pgvector_version=pgvector_version,
        error=None if pgvector_version is not None else "pgvector_no_habilitado",
    )
