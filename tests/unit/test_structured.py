from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from spark_rag.config import Settings
from spark_rag.rag.chain import SparkRagChain
from spark_rag.rag.errors import (
    EmptyModelResponse,
    LLMConfigurationError,
    LLMUnavailable,
)
from spark_rag.rag.mood import Mood
from spark_rag.rag.sources import RetrievedChunk
from spark_rag.rag.structured import (
    FORMAT_INSTRUCTION,
    TOOL_NAME,
    SparkOutput,
    build_structured_generator,
    clean_model_text,
    parse_generation,
    parse_json_reply,
    parse_prompt_message,
    parse_tool_message,
)

JSON_REPLY = '{"reply": "Saludos, Reclamador.", "mood": "enthusiastic"}'


# --- Limpieza del texto ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"<think>analizo la petición</think>{JSON_REPLY}", JSON_REPLY),
        (f"<THINK>\nvarias\nlíneas\n</THINK>\n{JSON_REPLY}", JSON_REPLY),
        (f"razonamiento sin apertura</think>{JSON_REPLY}", JSON_REPLY),
        (f"```json\n{JSON_REPLY}\n```", JSON_REPLY),
        (f"```\n{JSON_REPLY}\n```", JSON_REPLY),
        ("Texto normal.", "Texto normal."),
    ],
)
def test_limpia_razonamiento_y_vallas(text: str, expected: str) -> None:
    assert clean_model_text(text) == expected


def test_razonamiento_truncado_no_deja_texto() -> None:
    assert clean_model_text("<think>me quedé sin tokens pensando en") == ""


# --- Extracción del JSON --------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        JSON_REPLY,
        f"Aquí tienes: {JSON_REPLY} Fin.",
        '{"reply": roto} ' + JSON_REPLY,
    ],
)
def test_extrae_el_primer_json_valido(text: str) -> None:
    assert parse_json_reply(text) == {
        "reply": "Saludos, Reclamador.",
        "mood": "enthusiastic",
    }


@pytest.mark.parametrize(
    "text",
    [
        '{"reply": "sin cerrar"',
        '{"mood": "idle"}',
        '{"reply": "   ", "mood": "idle"}',
        '["reply", "mood"]',
        "Sin JSON.",
    ],
)
def test_json_ausente_o_invalido(text: str) -> None:
    assert parse_json_reply(text) is None


# --- Modo prompt ----------------------------------------------------------


def test_modo_prompt_ignora_reasoning_content() -> None:
    message = AIMessage(
        content=f"<think>plan secreto</think>```json\n{JSON_REPLY}\n```",
        additional_kwargs={"reasoning_content": "razonamiento oculto"},
    )
    reply, mood, structured = parse_generation(parse_prompt_message(message))

    assert reply == "Saludos, Reclamador."
    assert mood is Mood.ENTHUSIASTIC
    assert structured is True
    assert "plan secreto" not in reply
    assert "razonamiento oculto" not in reply


def test_modo_prompt_ignora_partes_de_razonamiento() -> None:
    message = AIMessage(
        content=[
            {"type": "reasoning", "text": "razonamiento oculto"},
            {"type": "text", "text": JSON_REPLY},
        ]
    )
    reply, _, structured = parse_generation(parse_prompt_message(message))
    assert reply == "Saludos, Reclamador."
    assert structured is True


def test_modo_prompt_mood_invalido_se_normaliza() -> None:
    message = AIMessage(content='{"reply": "Hola.", "mood": "furioso"}')
    reply, mood, structured = parse_generation(parse_prompt_message(message))
    assert (reply, mood, structured) == ("Hola.", Mood.THINKING, True)


def test_modo_prompt_sin_json_usa_el_texto_limpio() -> None:
    message = AIMessage(content="<think>x</think>Mis protocolos no registran eso.")
    reply, mood, structured = parse_generation(parse_prompt_message(message))
    assert (reply, mood, structured) == (
        "Mis protocolos no registran eso.",
        Mood.THINKING,
        False,
    )


@pytest.mark.parametrize("content", ["", "   ", "<think>solo razonamiento</think>"])
def test_modo_prompt_texto_vacio_es_empty_model_response(content: str) -> None:
    with pytest.raises(EmptyModelResponse):
        parse_prompt_message(AIMessage(content=content))


def test_generador_prompt_anade_la_instruccion_al_ultimo_turno() -> None:
    seen: list[list[BaseMessage]] = []

    def fake_llm(messages: list[BaseMessage]) -> AIMessage:
        seen.append(messages)
        return AIMessage(content=JSON_REPLY)

    generator = build_structured_generator(RunnableLambda(fake_llm), "prompt")
    output = generator.invoke(
        [SystemMessage(content="sistema"), HumanMessage(content="pregunta")]
    )

    sent = seen[0]
    assert sent[0].content == "sistema"
    assert sent[-1].content == f"pregunta\n\n{FORMAT_INSTRUCTION}"
    assert parse_generation(output) == ("Saludos, Reclamador.", Mood.ENTHUSIASTIC, True)


# --- Modo function_calling --------------------------------------------------


def _tool_message(args: dict[str, Any], name: str = TOOL_NAME, content: str = ""):
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": "call_1", "type": "tool_call"}],
    )


def test_tool_call_bien_formado() -> None:
    output = parse_tool_message(_tool_message({"reply": "Hola.", "mood": "idle"}))

    assert isinstance(output["parsed"], SparkOutput)
    assert parse_generation(output) == ("Hola.", Mood.IDLE, True)


def test_tool_call_con_mood_invalido_rescata_el_texto() -> None:
    output = parse_tool_message(_tool_message({"reply": "Hola.", "mood": "furioso"}))

    assert output["parsed"] is None
    assert parse_generation(output) == ("Hola.", Mood.THINKING, False)


def test_tool_call_de_otra_herramienta_usa_el_texto() -> None:
    message = _tool_message({"reply": "x"}, name="Otra", content=JSON_REPLY)
    reply, _, structured = parse_generation(parse_tool_message(message))
    assert (reply, structured) == ("Saludos, Reclamador.", True)


@pytest.mark.parametrize(
    "message",
    [
        AIMessage(content=""),
        _tool_message({"reply": "   ", "mood": "idle"}),
        _tool_message({"mood": "idle"}),
    ],
)
def test_sin_tool_call_util_ni_texto_es_empty_model_response(
    message: AIMessage,
) -> None:
    with pytest.raises(EmptyModelResponse):
        parse_tool_message(message)


def test_generador_function_calling_fuerza_la_herramienta() -> None:
    calls: list[dict[str, Any]] = []

    def bind_tools(tools: list[Any], **kwargs: Any) -> RunnableLambda:
        calls.append({"tools": tools, **kwargs})
        return RunnableLambda(
            lambda _: _tool_message({"reply": "Hola.", "mood": "idle"})
        )

    generator = build_structured_generator(
        SimpleNamespace(bind_tools=bind_tools), "function_calling"
    )
    output = generator.invoke([HumanMessage(content="hola")])

    assert calls == [{"tools": [SparkOutput], "tool_choice": TOOL_NAME}]
    assert parse_generation(output) == ("Hola.", Mood.IDLE, True)


def test_peticion_function_calling_sin_parallel_tool_calls() -> None:
    from spark_rag.rag.llm import build_chat_model

    model = build_chat_model(
        Settings(
            _env_file=None,
            llm_provider="openai_compat",
            openai_compat_api_key="nvapi-FAKEFAKEFAKEFAKE",
            openai_compat_chat_model="modelo-falso",
        )
    )
    bound = build_structured_generator(model, "function_calling").first
    payload = model._get_request_payload([HumanMessage(content="hola")], **bound.kwargs)

    # Misma petición que la prueba de humo: herramienta forzada, nada más.
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": TOOL_NAME},
    }
    assert [tool["function"]["name"] for tool in payload["tools"]] == [TOOL_NAME]
    assert "parallel_tool_calls" not in payload


def test_modo_no_soportado() -> None:
    with pytest.raises(LLMConfigurationError, match="no está soportado"):
        build_structured_generator(RunnableLambda(lambda _: None), "json_mode")


# --- Integración con la cadena ------------------------------------------------


CHUNK = RetrievedChunk(
    content="Chakas fue un humano de Erde-Tyrene.",
    similarity=0.8,
    metadata={"source": "history/dossier.pdf", "category": "history", "page": 3},
)


def _chain_with(content: str) -> SparkRagChain:
    generator = build_structured_generator(
        RunnableLambda(lambda _: AIMessage(content=content)), "prompt"
    )
    return SparkRagChain(retrieve=lambda _: [CHUNK], generator=generator)


def test_cadena_con_modo_prompt() -> None:
    result = _chain_with(f"<think>x</think>{JSON_REPLY}").invoke("¿Quién eres?")
    assert (result.reply, result.mood, result.structured) == (
        "Saludos, Reclamador.",
        Mood.ENTHUSIASTIC,
        True,
    )


def test_cadena_respuesta_vacia_es_llm_unavailable() -> None:
    with pytest.raises(LLMUnavailable):
        _chain_with("<think>sin tokens").invoke("¿Quién eres?")
