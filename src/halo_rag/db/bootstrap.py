"""Inicialización de la base de datos vectorial.

Responsabilidad:
    Asegurar la extensión `vector` y aplicar `schema.sql` de forma idempotente.
"""

from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def initialize_database() -> None:
    """Habilita pgvector y crea las tablas/índices del esquema.

    Raises:
        NotImplementedError: Pendiente de implementación en la Fase 2.
    """
    raise NotImplementedError("Inicialización de la base pendiente (Fase 2).")
