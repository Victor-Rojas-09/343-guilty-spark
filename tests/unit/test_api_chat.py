"""Pruebas de `POST /api/chat` con dobles (sin PostgreSQL ni Gemini)."""

from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from spark_rag.api.chat import get_service
from spark_rag.api.main import app
from spark_rag.config import Settings
from spark_rag.rag.errors import (
    LLMConfigurationError,
    LLMUnavailable,
    RetrievalUnavailable,
)
from spark_rag.rag.service import ChatService
from tests.unit.fakes import FakeChain, InMemoryStore

SECRET = "clave-super-secreta"


@pytest.fixture
def fakes():
    store, chain = InMemoryStore(), FakeChain()
    service = ChatService(
        chain=chain,
        store=store,
        settings=Settings(_env_file=None),
        model_name="modelo-test",
    )
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app, raise_server_exceptions=False)
    yield client, store, chain
    app.dependency_overrides.clear()


def _assert_controlled_error(response, status_code: int, error: str) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert set(body) == {"error", "detail"}
    assert body["error"] == error
    text = response.text
    for leak in ("Traceback", "postgresql://", SECRET, "secret@"):
        assert leak not in text


def test_contrato_json_completo(fakes) -> None:
    client, _, _ = fakes
    response = client.post("/api/chat", json={"message": "¿Quién era Chakas?"})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["reply"], str) and body["reply"]
    assert body["mood"] in {"idle", "thinking", "enthusiastic", "rampant"}
    assert body["sources"] == ["history/dossier.pdf"]
    assert all(isinstance(source, str) for source in body["sources"])
    assert body["audio_url"] is None
    assert UUID(body["session_id"])
    detail = body["source_details"][0]
    assert detail["source"] == "history/dossier.pdf"
    assert detail["page"] == 3
    assert detail["source_class"] == "canon"


def test_misma_sesion_conserva_contexto(fakes) -> None:
    client, _, chain = fakes
    first = client.post("/api/chat", json={"message": "Háblame de Chakas"}).json()
    second = client.post(
        "/api/chat",
        json={"message": "¿Y después?", "session_id": first["session_id"]},
    )

    assert second.status_code == 200
    assert second.json()["session_id"] == first["session_id"]
    history = chain.calls[1]["history"]
    assert history[0].content == "Háblame de Chakas"


def test_sesiones_distintas_aisladas(fakes) -> None:
    client, _, chain = fakes
    a = client.post("/api/chat", json={"message": "Dato privado de A"}).json()
    b = client.post("/api/chat", json={"message": "Hola desde B"}).json()
    client.post("/api/chat", json={"message": "Sigue", "session_id": b["session_id"]})

    assert a["session_id"] != b["session_id"]
    history_b = [t.content for t in chain.calls[2]["history"]]
    assert all("privado de A" not in content for content in history_b)


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {},
        {"session_id": str(uuid4())},
        {"message": "a" * 2001},
        {"message": "Hola", "session_id": "no-es-un-uuid"},
        {"message": 123},
    ],
)
def test_entradas_invalidas(fakes, payload) -> None:
    client, store, chain = fakes
    response = client.post("/api/chat", json=payload)

    _assert_controlled_error(response, 422, "invalid_request")
    assert chain.calls == []
    assert store.messages == {}


def test_cuerpo_no_json(fakes) -> None:
    client, _, _ = fakes
    response = client.post(
        "/api/chat",
        content="esto no es json",
        headers={"Content-Type": "application/json"},
    )
    _assert_controlled_error(response, 422, "invalid_request")


def test_sesion_inexistente(fakes) -> None:
    client, _, _ = fakes
    response = client.post(
        "/api/chat", json={"message": "Hola", "session_id": str(uuid4())}
    )
    _assert_controlled_error(response, 404, "session_not_found")


def test_fallo_de_gemini(fakes) -> None:
    client, store, chain = fakes
    chain.error = LLMUnavailable(f"quota exceeded {SECRET}")
    response = client.post("/api/chat", json={"message": "Hola"})

    _assert_controlled_error(response, 503, "llm_unavailable")
    assert store.messages == {}


def test_fallo_de_recuperacion(fakes) -> None:
    client, _, chain = fakes
    chain.error = RetrievalUnavailable("postgresql://user:secret@db")
    response = client.post("/api/chat", json={"message": "Hola"})
    _assert_controlled_error(response, 503, "retrieval_unavailable")


def test_fallo_de_base_de_datos(fakes) -> None:
    client, store, _ = fakes
    store.fail = True
    response = client.post("/api/chat", json={"message": "Hola"})
    _assert_controlled_error(response, 503, "storage_unavailable")


def test_error_inesperado_no_expone_trazas(fakes) -> None:
    client, _, chain = fakes
    chain.error = RuntimeError(f"fallo con {SECRET}")
    response = client.post("/api/chat", json={"message": "Hola"})
    _assert_controlled_error(response, 500, "internal_error")


def test_llm_sin_configurar() -> None:
    def not_configured():
        raise LLMConfigurationError("GEMINI_API_KEY no está configurada.")

    app.dependency_overrides[get_service] = not_configured
    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/chat", json={"message": "Hola"})
    finally:
        app.dependency_overrides.clear()

    _assert_controlled_error(response, 503, "llm_not_configured")


def test_health_sigue_disponible(monkeypatch) -> None:
    from spark_rag.api import main
    from spark_rag.db.connection import DatabaseStatus

    monkeypatch.setattr(
        main,
        "check_database",
        lambda: DatabaseStatus(ok=True, server_version="16", pgvector_version="0.8.0"),
    )
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_arranque_avisa_si_falta_la_clave_del_respaldo(monkeypatch, caplog) -> None:
    import logging

    from spark_rag.api import main

    message = (
        "El respaldo del chat 'openai_compat' (LLM_FALLBACK_PROVIDERS) no está "
        "configurado. OPENAI_COMPAT_API_KEY no está configurada."
    )
    monkeypatch.setattr(
        "spark_rag.rag.fallback.chat_configuration_problem", lambda: message
    )
    with caplog.at_level(logging.ERROR, logger="spark_rag.api.main"):
        main.check_chat_configuration()

    assert f"Chat sin configurar: {message}" in caplog.text


def test_arranque_sin_problemas_no_registra_error(monkeypatch, caplog) -> None:
    import logging

    from spark_rag.api import main

    monkeypatch.setattr(
        "spark_rag.rag.fallback.chat_configuration_problem", lambda: None
    )
    with caplog.at_level(logging.ERROR, logger="spark_rag.api.main"):
        main.check_chat_configuration()

    assert "Chat sin configurar" not in caplog.text


def test_la_comprobacion_del_arranque_nunca_lo_impide(monkeypatch, caplog) -> None:
    from spark_rag.api import main

    def broken() -> None:
        raise RuntimeError("fallo inesperado")

    monkeypatch.setattr("spark_rag.rag.fallback.chat_configuration_problem", broken)
    main.check_chat_configuration()  # no lanza

    assert "No se pudo comprobar la configuración del chat" in caplog.text
