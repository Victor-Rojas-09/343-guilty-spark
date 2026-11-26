from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from spark_rag.rag.chain import SparkOutput, SparkRagChain, parse_generation
from spark_rag.rag.errors import LLMUnavailable, RetrievalUnavailable
from spark_rag.rag.history import ChatTurn
from spark_rag.rag.mood import Mood
from spark_rag.rag.prompts import EVIDENCE_MISSING
from spark_rag.rag.sources import RetrievedChunk

CHUNK = RetrievedChunk(
    content="Chakas fue un humano de Erde-Tyrene.",
    similarity=0.8,
    metadata={"source": "history/dossier.pdf", "category": "history", "page": 3},
)


class _Recorder:
    """Generador falso que guarda los mensajes recibidos."""

    def __init__(self, output: Any) -> None:
        self.output = output
        self.messages: list[BaseMessage] = []

    def __call__(self, messages: list[BaseMessage]) -> Any:
        self.messages = messages
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def _chain(retrieved: list[RetrievedChunk], output: Any):
    queries: list[str] = []

    def retrieve(query: str) -> list[RetrievedChunk]:
        queries.append(query)
        return retrieved

    recorder = _Recorder(output)
    chain = SparkRagChain(retrieve=retrieve, generator=RunnableLambda(recorder))
    return chain, recorder, queries


def test_ruta_con_evidencia() -> None:
    chain, recorder, queries = _chain(
        [CHUNK], SparkOutput(reply="Mi patrón deriva de Chakas.", mood="thinking")
    )
    result = chain.invoke("¿Quién era Chakas?")

    assert queries == ["¿Quién era Chakas?"]
    assert result.reply == "Mi patrón deriva de Chakas."
    assert result.mood is Mood.THINKING
    assert result.structured
    assert result.sources == ["history/dossier.pdf"]
    assert result.source_details[0]["page"] == 3
    assert isinstance(recorder.messages[0], SystemMessage)
    assert "Erde-Tyrene" in recorder.messages[-1].content
    assert EVIDENCE_MISSING not in recorder.messages[-1].content


def test_ruta_sin_evidencia() -> None:
    chain, recorder, _ = _chain(
        [], SparkOutput(reply="Mis protocolos no registran datos.", mood="thinking")
    )
    result = chain.invoke("¿Quién es Cortana?")

    assert not result.has_evidence
    assert result.sources == []
    assert result.source_details == []
    assert EVIDENCE_MISSING in recorder.messages[-1].content


def test_historial_y_consulta_de_recuperacion() -> None:
    chain, recorder, queries = _chain([CHUNK], SparkOutput(reply="Sí.", mood="idle"))
    chain.invoke(
        "¿Y después?",
        history=[
            ChatTurn(role="user", content="Háblame de Chakas"),
            ChatTurn(role="assistant", content="Fue mi origen."),
        ],
        retrieval_query="Háblame de Chakas\n¿Y después?",
    )

    assert queries == ["Háblame de Chakas\n¿Y después?"]
    assert isinstance(recorder.messages[1], HumanMessage)
    assert recorder.messages[1].content == "Háblame de Chakas"
    assert isinstance(recorder.messages[2], AIMessage)


def test_salida_include_raw_parseada() -> None:
    output = {
        "raw": AIMessage(content=""),
        "parsed": SparkOutput(reply="¡Fascinante!", mood="enthusiastic"),
        "parsing_error": None,
    }
    chain, _, _ = _chain([CHUNK], output)
    result = chain.invoke("x")

    assert result.reply == "¡Fascinante!"
    assert result.mood is Mood.ENTHUSIASTIC


def test_fallo_de_parseo_usa_texto_crudo_y_mood_por_defecto() -> None:
    output = {
        "raw": AIMessage(content="Respuesta sin esquema."),
        "parsed": None,
        "parsing_error": ValueError("schema"),
    }
    chain, _, _ = _chain([CHUNK], output)
    result = chain.invoke("x")

    assert result.reply == "Respuesta sin esquema."
    assert result.mood is Mood.THINKING
    assert not result.structured


def test_mood_invalido_en_tool_call_se_normaliza() -> None:
    raw = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "SparkOutput",
                "args": {"reply": "Hola, Reclamador.", "mood": "furioso"},
                "id": "1",
            }
        ],
    )
    reply, mood, structured = parse_generation(
        {"raw": raw, "parsed": None, "parsing_error": ValueError()}
    )

    assert reply == "Hola, Reclamador."
    assert mood is Mood.THINKING
    assert not structured


def test_contenido_por_partes() -> None:
    reply, mood, _ = parse_generation(
        AIMessage(content=[{"type": "text", "text": "Parte 1. "}, "Parte 2."])
    )
    assert reply == "Parte 1. Parte 2."
    assert mood is Mood.THINKING


def test_fallo_del_modelo_es_llm_unavailable() -> None:
    chain, _, _ = _chain([CHUNK], RuntimeError("quota exceeded key=abc"))
    with pytest.raises(LLMUnavailable) as info:
        chain.invoke("x")
    assert "abc" not in str(info.value)


def test_respuesta_vacia_es_llm_unavailable() -> None:
    chain, _, _ = _chain([CHUNK], SparkOutput(reply="   ", mood="idle"))
    with pytest.raises(LLMUnavailable):
        chain.invoke("x")


def test_fallo_de_recuperacion_es_retrieval_unavailable() -> None:
    def broken(_: str) -> list[RetrievedChunk]:
        raise ConnectionError("postgresql://user:secret@db")

    chain = SparkRagChain(
        retrieve=broken,
        generator=RunnableLambda(lambda _: SparkOutput(reply="x", mood="idle")),
    )
    with pytest.raises(RetrievalUnavailable) as info:
        chain.invoke("x")
    assert "secret" not in str(info.value)
