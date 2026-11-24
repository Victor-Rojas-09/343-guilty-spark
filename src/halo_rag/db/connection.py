"""Conexión a PostgreSQL con soporte de pgvector.

Responsabilidad:
    Abrir conexiones usando la configuración central y registrar el adaptador de
    `pgvector` para psycopg. Toda la aplicación debe obtener conexiones aquí.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from halo_rag.config import get_settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Abre una conexión a PostgreSQL y la cierra al salir del contexto.

    Yields:
        Conexión de psycopg lista para ejecutar consultas.
    """
    settings = get_settings()
    with psycopg.connect(settings.database_url) as connection:
        # TODO(Fase 2): register_vector(connection) para (de)serializar embeddings.
        yield connection
