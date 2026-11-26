"""Persistencia de sesiones de conversación y sus mensajes (MEM-01, MEM-02).

Tablas (manual §5.1):

* `chat_sessions`: `id` (UUID), fecha de creación y última actividad.
* `chat_messages`: rol, contenido, fecha, `mood`, fuentes usadas, versión del
  prompt y modelo que generó la respuesta.

Todas las lecturas filtran por `session_id`, de modo que el historial de una
sesión nunca se mezcla con otra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from spark_rag.db.connection import get_connection

SESSION_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id                UUID PRIMARY KEY,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_activity_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id              BIGSERIAL PRIMARY KEY,
        session_id      UUID NOT NULL
                        REFERENCES chat_sessions (id) ON DELETE CASCADE,
        role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content         TEXT NOT NULL,
        mood            TEXT,
        sources         JSONB NOT NULL DEFAULT '[]'::jsonb,
        prompt_version  TEXT,
        model           TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS chat_messages_session_idx
        ON chat_messages (session_id, id)
    """,
)


def initialize_session_schema(connection: psycopg.Connection) -> None:
    """Crea las tablas de sesiones de forma idempotente."""
    for statement in SESSION_SCHEMA_STATEMENTS:
        connection.execute(statement)


def session_exists(connection: psycopg.Connection, session_id: UUID) -> bool:
    row = connection.execute(
        "SELECT 1 AS found FROM chat_sessions WHERE id = %s",
        (session_id,),
    ).fetchone()
    return row is not None


def get_session(
    connection: psycopg.Connection, session_id: UUID
) -> dict[str, Any] | None:
    return connection.execute(
        """
        SELECT id, created_at, last_activity_at
        FROM chat_sessions
        WHERE id = %s
        """,
        (session_id,),
    ).fetchone()


def get_recent_messages(
    connection: psycopg.Connection, session_id: UUID, *, limit: int
) -> list[dict[str, Any]]:
    """Últimos `limit` mensajes de la sesión, en orden cronológico."""
    if limit <= 0:
        return []
    rows = connection.execute(
        """
        SELECT id, role, content, mood, sources, prompt_version, model, created_at
        FROM chat_messages
        WHERE session_id = %s
        ORDER BY id DESC
        LIMIT %s
        """,
        (session_id, limit),
    ).fetchall()
    return list(reversed(rows))


def save_exchange(
    connection: psycopg.Connection,
    *,
    session_id: UUID,
    user_message: str,
    assistant_message: str,
    mood: str,
    sources: Sequence[dict[str, Any]],
    prompt_version: str,
    model: str,
) -> None:
    """Guarda un turno completo (usuario + asistente) y actualiza la sesión.

    Crea la sesión si no existía. Debe ejecutarse dentro de una única
    transacción para no dejar turnos a medias.
    """
    connection.execute(
        """
        INSERT INTO chat_sessions (id) VALUES (%s)
        ON CONFLICT (id) DO UPDATE SET last_activity_at = now()
        """,
        (session_id,),
    )
    connection.execute(
        """
        INSERT INTO chat_messages (session_id, role, content)
        VALUES (%s, 'user', %s)
        """,
        (session_id, user_message),
    )
    connection.execute(
        """
        INSERT INTO chat_messages (
            session_id, role, content, mood, sources, prompt_version, model
        )
        VALUES (%s, 'assistant', %s, %s, %s, %s, %s)
        """,
        (
            session_id,
            assistant_message,
            mood,
            Jsonb(list(sources)),
            prompt_version,
            model,
        ),
    )


class PostgresSessionStore:
    """Repositorio de sesiones sobre PostgreSQL usado por el servicio de chat."""

    def initialize(self) -> None:
        with get_connection() as connection:
            initialize_session_schema(connection)

    def exists(self, session_id: UUID) -> bool:
        with get_connection() as connection:
            return session_exists(connection, session_id)

    def recent_messages(self, session_id: UUID, limit: int) -> list[dict[str, Any]]:
        with get_connection() as connection:
            return get_recent_messages(connection, session_id, limit=limit)

    def save_exchange(self, **kwargs: Any) -> None:
        with get_connection() as connection:
            save_exchange(connection, **kwargs)
