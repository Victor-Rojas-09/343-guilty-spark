"""Salida estructurada `{reply, mood}` del modelo de chat.

- Gemini: `with_structured_output(SparkOutput, include_raw=True)` (ver
  `chain.build_generator`).
- Proveedor compatible con OpenAI (`OPENAI_COMPAT_STRUCTURED_MODE`):
  - `function_calling`: `bind_tools([SparkOutput], tool_choice="SparkOutput")`.
    No se usa `with_structured_output` porque añade `parallel_tool_calls`, que
    no se ha verificado con el proveedor; esta petición sí (prueba de humo).
  - `prompt`: instrucción de formato JSON en el último turno y parser tolerante.

En ambos modos el razonamiento del modelo (`<think>…</think>`,
`reasoning_content`, partes de tipo «reasoning») nunca llega al `reply`. Si tras
limpiar no queda texto útil se lanza `EmptyModelResponse`, que activa el
proveedor de respaldo.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field, ValidationError

from spark_rag.config import STRUCTURED_MODES
from spark_rag.rag.errors import EmptyModelResponse, LLMConfigurationError
from spark_rag.rag.mood import DEFAULT_MOOD, Mood, normalize_mood


class SparkOutput(BaseModel):
    """Esquema que se exige al modelo de chat."""

    reply: str = Field(
        description="Respuesta de 343 Guilty Spark en primera persona y en personaje."
    )
    mood: Mood = Field(
        description="Estado de ánimo de la respuesta: idle, thinking, "
        "enthusiastic o rampant."
    )


TOOL_NAME = SparkOutput.__name__

FORMAT_INSTRUCTION = (
    "Formato de salida obligatorio: responde únicamente con un objeto JSON con "
    'dos claves: "reply" (tu respuesta en personaje, como texto) y "mood" (uno '
    "de: idle, thinking, enthusiastic, rampant). No escribas nada fuera del JSON."
)

_REASONING_PART_TYPES = frozenset({"reasoning", "thinking", "reasoning_content"})
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _text_from_message(message: Any) -> str:
    """Texto de un `AIMessage` (plano o por partes), sin partes de razonamiento."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in _REASONING_PART_TYPES:
                    continue
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return ""


def clean_model_text(text: str) -> str:
    """Quita el razonamiento `<think>` y las vallas de código, conservando el resto."""
    text = _THINK_BLOCK.sub("", text)
    # Cierre sin apertura (la plantilla abrió el bloque): lo útil va después.
    if _THINK_CLOSE.search(text):
        text = _THINK_CLOSE.split(text)[-1]
    # Apertura sin cierre (razonamiento truncado): todo lo posterior es razonamiento.
    if _THINK_OPEN.search(text):
        text = _THINK_OPEN.split(text)[0]
    text = _FENCE.sub(lambda match: match.group(1), text)
    return text.strip()


def parse_json_reply(text: str) -> dict[str, Any] | None:
    """Primer objeto JSON con `reply` no vacío del texto ya limpio, o None."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        reply = value.get("reply") if isinstance(value, dict) else None
        if isinstance(reply, str) and reply.strip():
            return {"reply": reply, "mood": value.get("mood")}
    return None


def _clean_message(message: Any) -> AIMessage:
    return AIMessage(content=clean_model_text(_text_from_message(message)))


def parse_tool_message(message: Any) -> dict[str, Any]:
    """Salida del modo `function_calling` en formato `include_raw`.

    Devuelve `{"raw", "parsed", "parsing_error"}` como `with_structured_output`,
    para que `parse_generation` la trate igual que la de Gemini.
    """
    for call in getattr(message, "tool_calls", None) or []:
        if call.get("name") != TOOL_NAME:
            continue
        args = call.get("args") or {}
        try:
            parsed = SparkOutput.model_validate(args)
        except ValidationError as exc:
            # El texto puede ser válido aunque el mood no: lo rescata parse_generation.
            if isinstance(args.get("reply"), str) and args["reply"].strip():
                return {"raw": message, "parsed": None, "parsing_error": exc}
            continue
        if parsed.reply.strip():
            return {"raw": message, "parsed": parsed, "parsing_error": None}

    # Sin tool call válido: se intenta el texto (puede traer el JSON en línea).
    return parse_prompt_message(message)


def parse_prompt_message(message: Any) -> dict[str, Any]:
    """Salida del modo `prompt` (o texto libre) en formato `include_raw`."""
    cleaned = _clean_message(message)
    parsed = parse_json_reply(cleaned.content)
    if parsed is not None:
        return {"raw": cleaned, "parsed": parsed, "parsing_error": None}
    if not cleaned.content:
        raise EmptyModelResponse("El modelo no devolvió texto útil.")
    return {"raw": cleaned, "parsed": None, "parsing_error": None}


def add_format_instruction(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Añade la instrucción de formato JSON al último turno del usuario."""
    if not messages or not isinstance(messages[-1], HumanMessage):
        return [*messages, HumanMessage(content=FORMAT_INSTRUCTION)]
    last = messages[-1]
    return [
        *messages[:-1],
        HumanMessage(content=f"{last.content}\n\n{FORMAT_INSTRUCTION}"),
    ]


def build_structured_generator(
    llm: BaseChatModel | Runnable[Any, Any], mode: str
) -> Runnable[list[BaseMessage], Any]:
    """Generador `{reply, mood}` para el proveedor compatible con OpenAI."""
    mode = mode.strip().lower()
    if mode == "function_calling":
        bound = llm.bind_tools([SparkOutput], tool_choice=TOOL_NAME)
        return bound | RunnableLambda(parse_tool_message)
    if mode == "prompt":
        return (
            RunnableLambda(add_format_instruction)
            | llm
            | RunnableLambda(parse_prompt_message)
        )
    raise LLMConfigurationError(
        f"OPENAI_COMPAT_STRUCTURED_MODE={mode!r} no está soportado. "
        f"Opciones: {', '.join(STRUCTURED_MODES)}."
    )


def parse_generation(output: Any) -> tuple[str, Mood, bool]:
    """Normaliza la salida del modelo a `(reply, mood, structured)`.

    Acepta un `SparkOutput`, un diccionario `include_raw` de LangChain
    (`raw`/`parsed`/`parsing_error`) o un mensaje de chat simple.
    """
    if isinstance(output, SparkOutput):
        return output.reply.strip(), normalize_mood(output.mood), True

    if isinstance(output, dict) and "raw" in output:
        parsed = output.get("parsed")
        if isinstance(parsed, SparkOutput):
            return parsed.reply.strip(), normalize_mood(parsed.mood), True
        if isinstance(parsed, dict) and parsed.get("reply"):
            return (
                str(parsed["reply"]).strip(),
                normalize_mood(parsed.get("mood")),
                True,
            )

        raw = output.get("raw")
        for call in getattr(raw, "tool_calls", None) or []:
            args = call.get("args") or {}
            if args.get("reply"):
                return (
                    str(args["reply"]).strip(),
                    normalize_mood(args.get("mood")),
                    False,
                )
        return _text_from_message(raw).strip(), DEFAULT_MOOD, False

    return _text_from_message(output).strip(), DEFAULT_MOOD, False
