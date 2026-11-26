"""Persistencia de sesiones contra PostgreSQL real (sin Gemini)."""

import os
import uuid

import pytest

from spark_rag.db.connection import get_connection
from spark_rag.db.sessions import (
    PostgresSessionStore,
    get_session,
    initialize_session_schema,
)

pytestmark = pytest.mark.integration

requires_database = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="Define RUN_INTEGRATION=1 con PostgreSQL/pgvector disponible.",
)


@pytest.fixture
def store():
    store = PostgresSessionStore()
    store.initialize()
    created: list[uuid.UUID] = []
    yield store, created
    with get_connection() as connection:
        for session_id in created:
            connection.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))


def _save(store: PostgresSessionStore, session_id: uuid.UUID, index: int) -> None:
    store.save_exchange(
        session_id=session_id,
        user_message=f"pregunta {index}",
        assistant_message=f"respuesta {index}",
        mood="thinking",
        sources=[{"source": "history/a.pdf", "page": index}],
        prompt_version="spark-test",
        model="modelo-test",
    )


@requires_database
def test_esquema_idempotente() -> None:
    with get_connection() as connection:
        initialize_session_schema(connection)
        initialize_session_schema(connection)
        row = connection.execute(
            "SELECT to_regclass('public.chat_messages') AS table_name"
        ).fetchone()
    assert row["table_name"] is not None


@requires_database
def test_crea_sesion_y_conserva_orden(store) -> None:
    store_, created = store
    session_id = uuid.uuid4()
    created.append(session_id)

    assert not store_.exists(session_id)
    for index in range(3):
        _save(store_, session_id, index)
    assert store_.exists(session_id)

    messages = store_.recent_messages(session_id, limit=10)
    assert [m["content"] for m in messages] == [
        "pregunta 0",
        "respuesta 0",
        "pregunta 1",
        "respuesta 1",
        "pregunta 2",
        "respuesta 2",
    ]
    assistant = messages[-1]
    assert assistant["role"] == "assistant"
    assert assistant["mood"] == "thinking"
    assert assistant["sources"] == [{"source": "history/a.pdf", "page": 2}]
    assert assistant["prompt_version"] == "spark-test"
    assert assistant["model"] == "modelo-test"


@requires_database
def test_limite_devuelve_los_mas_recientes(store) -> None:
    store_, created = store
    session_id = uuid.uuid4()
    created.append(session_id)
    for index in range(4):
        _save(store_, session_id, index)

    messages = store_.recent_messages(session_id, limit=2)
    assert [m["content"] for m in messages] == ["pregunta 3", "respuesta 3"]
    assert store_.recent_messages(session_id, limit=0) == []


@requires_database
def test_actualiza_ultima_actividad(store) -> None:
    store_, created = store
    session_id = uuid.uuid4()
    created.append(session_id)

    _save(store_, session_id, 0)
    with get_connection() as connection:
        first = get_session(connection, session_id)
    _save(store_, session_id, 1)
    with get_connection() as connection:
        second = get_session(connection, session_id)

    assert second["created_at"] == first["created_at"]
    assert second["last_activity_at"] >= first["last_activity_at"]


@requires_database
def test_sesiones_aisladas(store) -> None:
    store_, created = store
    session_a, session_b = uuid.uuid4(), uuid.uuid4()
    created.extend([session_a, session_b])

    _save(store_, session_a, 1)
    _save(store_, session_b, 2)

    contents_a = {m["content"] for m in store_.recent_messages(session_a, limit=10)}
    contents_b = {m["content"] for m in store_.recent_messages(session_b, limit=10)}
    assert contents_a == {"pregunta 1", "respuesta 1"}
    assert contents_b == {"pregunta 2", "respuesta 2"}
