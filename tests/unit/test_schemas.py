import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from spark_rag.api.schemas import ChatRequest, ChatResponse, SourceDetail
from spark_rag.rag.mood import Mood


def test_solicitud_valida_recorta_espacios() -> None:
    request = ChatRequest(message="  ¿Quién eres?  ")
    assert request.message == "¿Quién eres?"
    assert request.session_id is None


def test_solicitud_acepta_session_id_uuid() -> None:
    session_id = uuid4()
    request = ChatRequest(message="Hola", session_id=str(session_id))
    assert request.session_id == session_id


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_rechaza_mensaje_vacio(message: str) -> None:
    with pytest.raises(ValidationError, match="vacío"):
        ChatRequest(message=message)


def test_rechaza_mensaje_ausente() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"session_id": str(uuid4())})


def test_rechaza_mensaje_demasiado_largo() -> None:
    with pytest.raises(ValidationError, match="máximo"):
        ChatRequest(message="a" * 2001)


@pytest.mark.parametrize("session_id", ["abc", "123", "session-uuid"])
def test_rechaza_session_id_mal_formado(session_id: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="Hola", session_id=session_id)


def test_respuesta_cumple_el_contrato_json() -> None:
    session_id = uuid4()
    response = ChatResponse(
        reply="Saludos, Reclamador.",
        mood="enthusiastic",
        sources=["history/01_Dossier.pdf"],
        session_id=session_id,
        source_details=[
            SourceDetail(
                source="history/01_Dossier.pdf",
                category="history",
                page=3,
                source_class="canon",
                similarity=0.82,
            )
        ],
    )
    payload = json.loads(response.model_dump_json())

    assert set(payload) >= {"reply", "mood", "sources", "audio_url", "session_id"}
    assert isinstance(payload["reply"], str)
    assert payload["mood"] == "enthusiastic"
    assert payload["sources"] == ["history/01_Dossier.pdf"]
    assert payload["audio_url"] is None
    assert payload["session_id"] == str(session_id)
    assert payload["source_details"][0]["page"] == 3


def test_respuesta_rechaza_mood_fuera_del_conjunto() -> None:
    with pytest.raises(ValidationError):
        ChatResponse(reply="x", mood="feliz", session_id=uuid4())


def test_respuesta_sin_fuentes_por_defecto() -> None:
    response = ChatResponse(reply="x", mood=Mood.THINKING, session_id=uuid4())
    assert response.sources == []
    assert response.source_details == []
