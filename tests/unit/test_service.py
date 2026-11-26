from typing import Any
from uuid import UUID, uuid4

import pytest

from spark_rag.config import Settings
from spark_rag.rag.errors import LLMUnavailable, SessionNotFound, StorageUnavailable
from spark_rag.rag.history import ChatTurn
from spark_rag.rag.mood import Mood
from spark_rag.rag.service import ChatService, build_retrieval_query
from tests.unit.fakes import FakeChain, InMemoryStore


def _service(chain: FakeChain, store: InMemoryStore, **settings: Any) -> ChatService:
    return ChatService(
        chain=chain,
        store=store,
        settings=Settings(_env_file=None, **settings),
        model_name="modelo-test",
        prompt_version="spark-test",
    )


def test_sin_session_id_crea_una_sesion_nueva() -> None:
    store, chain = InMemoryStore(), FakeChain()
    reply = _service(chain, store).chat("¿Quién era Chakas?")

    assert isinstance(reply.session_id, UUID)
    assert reply.reply == "Respuesta a: ¿Quién era Chakas?"
    assert reply.mood is Mood.THINKING
    assert reply.sources == ["history/dossier.pdf"]
    assert reply.source_details[0]["page"] == 3
    assert reply.audio_url is None
    assert chain.calls[0]["history"] == []

    saved = store.messages[reply.session_id]
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[1]["prompt_version"] == "spark-test"
    assert saved[1]["model"] == "modelo-test"
    assert saved[1]["sources"][0]["source"] == "history/dossier.pdf"


def test_session_id_desconocido() -> None:
    with pytest.raises(SessionNotFound):
        _service(FakeChain(), InMemoryStore()).chat("Hola", uuid4())


def test_turnos_consecutivos_conservan_historial() -> None:
    store, chain = InMemoryStore(), FakeChain()
    service = _service(chain, store)
    first = service.chat("Háblame de Chakas")
    second = service.chat("¿Y después?", first.session_id)

    assert second.session_id == first.session_id
    history = chain.calls[1]["history"]
    assert [t.content for t in history] == [
        "Háblame de Chakas",
        "Respuesta a: Háblame de Chakas",
    ]
    assert chain.calls[1]["retrieval_query"] == "Háblame de Chakas\n¿Y después?"
    assert len(store.messages[first.session_id]) == 4


def test_sesiones_distintas_no_comparten_historial() -> None:
    store, chain = InMemoryStore(), FakeChain()
    service = _service(chain, store)
    a = service.chat("Secreto de la sesión A")
    b = service.chat("Pregunta de B")
    service.chat("Sigue", b.session_id)

    assert a.session_id != b.session_id
    contents = [t.content for t in chain.calls[2]["history"]]
    assert all("sesión A" not in content for content in contents)


def test_historial_enviado_respeta_el_limite() -> None:
    store, chain = InMemoryStore(), FakeChain()
    service = _service(chain, store, chat_history_max_messages=2)
    session_id = service.chat("uno").session_id
    service.chat("dos", session_id)
    service.chat("tres", session_id)

    assert [t.content for t in chain.calls[2]["history"]] == [
        "dos",
        "Respuesta a: dos",
    ]


def test_fallo_del_modelo_no_persiste_nada() -> None:
    store, chain = InMemoryStore(), FakeChain()
    service = _service(chain, store)
    session_id = service.chat("uno").session_id
    chain.error = LLMUnavailable("El modelo de chat no respondió.")

    with pytest.raises(LLMUnavailable):
        service.chat("dos", session_id)
    assert len(store.messages[session_id]) == 2

    with pytest.raises(LLMUnavailable):
        service.chat("nueva sesión")
    assert len(store.messages) == 1


def test_sin_evidencia_devuelve_fuentes_vacias() -> None:
    reply = _service(FakeChain(chunks=[]), InMemoryStore()).chat("¿Quién es Cortana?")
    assert reply.sources == []
    assert reply.source_details == []


def test_fallo_de_base_de_datos_es_storage_unavailable() -> None:
    store = InMemoryStore()
    store.fail = True
    service = _service(FakeChain(), store)

    with pytest.raises(StorageUnavailable) as info:
        service.chat("Hola", uuid4())
    assert "secret" not in str(info.value)

    with pytest.raises(StorageUnavailable):
        service.chat("Hola")


def test_consulta_de_recuperacion() -> None:
    history = [
        ChatTurn(role="user", content="Primera"),
        ChatTurn(role="assistant", content="Respuesta"),
    ]
    assert build_retrieval_query(" Actual ", history) == "Primera\nActual"
    assert build_retrieval_query("Actual", []) == "Actual"
