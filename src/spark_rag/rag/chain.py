"""Cadena RAG de LangChain (RAG-01).

Flujo: consulta → embedding + recuperación en pgvector → contexto delimitado
ordenado por autoridad → System Prompt + historial → modelo de chat con salida
estructurada `{reply, mood}`.

La salida estructurada se pide con `include_raw=True`: si el modelo no respeta
el esquema, se recupera el texto crudo como `reply` y `mood` cae a `thinking`
en lugar de fallar la petición.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from spark_rag.rag.errors import (
    LLMUnavailable,
    RetrievalUnavailable,
    describe_error,
)
from spark_rag.rag.history import ChatTurn
from spark_rag.rag.mood import DEFAULT_MOOD, Mood, normalize_mood
from spark_rag.rag.prompts import build_messages
from spark_rag.rag.sources import (
    RetrievedChunk,
    format_context,
    public_sources,
    source_details,
)

logger = logging.getLogger(__name__)


class SparkOutput(BaseModel):
    """Esquema que se exige al modelo de chat."""

    reply: str = Field(
        description="Respuesta de 343 Guilty Spark en primera persona y en personaje."
    )
    mood: Mood = Field(
        description="Estado de ánimo de la respuesta: idle, thinking, "
        "enthusiastic o rampant."
    )


@dataclass(frozen=True)
class RagResult:
    """Resultado de un turno de la cadena RAG."""

    reply: str
    mood: Mood
    chunks: list[RetrievedChunk] = field(default_factory=list)
    structured: bool = True

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks)

    @property
    def sources(self) -> list[str]:
        return public_sources(self.chunks)

    @property
    def source_details(self) -> list[dict[str, Any]]:
        return source_details(self.chunks)


def _text_from_message(message: Any) -> str:
    """Extrae texto de un `AIMessage` (contenido plano o por partes)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ]
        return "".join(parts)
    return ""


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


class SparkRagChain:
    """Orquesta recuperación y generación con componentes inyectables."""

    def __init__(
        self,
        *,
        retrieve: Callable[[str], list[RetrievedChunk]],
        generator: Runnable[list[BaseMessage], Any],
    ) -> None:
        self._retrieve = retrieve
        self._generator = generator
        self._chain = (
            RunnablePassthrough.assign(chunks=RunnableLambda(self._safe_retrieve))
            | RunnablePassthrough.assign(messages=RunnableLambda(self._build_messages))
            | RunnableLambda(self._generate)
        ).with_config(run_name="spark_rag_chain")

    def _safe_retrieve(self, inputs: dict[str, Any]) -> list[RetrievedChunk]:
        try:
            return self._retrieve(inputs["retrieval_query"])
        except Exception as exc:  # se traduce a error de dominio
            logger.error("Fallo en la recuperación: %s", type(exc).__name__)
            raise RetrievalUnavailable(
                "La memoria documental no está disponible."
            ) from exc

    @staticmethod
    def _build_messages(inputs: dict[str, Any]) -> list[BaseMessage]:
        chunks: list[RetrievedChunk] = inputs["chunks"]
        return build_messages(
            question=inputs["question"],
            context=format_context(chunks),
            has_evidence=bool(chunks),
            history=inputs.get("history", ()),
        )

    def _generate(self, inputs: dict[str, Any]) -> RagResult:
        try:
            output = self._generator.invoke(inputs["messages"])
        except Exception as exc:  # se traduce a error de dominio
            logger.error("Fallo del modelo de chat: %s", describe_error(exc))
            raise LLMUnavailable("El modelo de chat no respondió.") from exc

        reply, mood, structured = parse_generation(output)
        if not reply:
            raise LLMUnavailable("El modelo de chat devolvió una respuesta vacía.")
        if not structured:
            logger.warning("Salida no estructurada del modelo; se usa mood=%s", mood)
        return RagResult(
            reply=reply,
            mood=mood,
            chunks=list(inputs["chunks"]),
            structured=structured,
        )

    def invoke(
        self,
        question: str,
        *,
        history: Sequence[ChatTurn] = (),
        retrieval_query: str | None = None,
    ) -> RagResult:
        """Ejecuta un turno completo de la cadena."""
        return self._chain.invoke(
            {
                "question": question,
                "retrieval_query": retrieval_query or question,
                "history": list(history),
            }
        )


def build_generator(llm: BaseChatModel) -> Runnable[list[BaseMessage], Any]:
    """Envuelve el modelo de chat para exigir el esquema `SparkOutput`."""
    return llm.with_structured_output(SparkOutput, include_raw=True)


def build_chain(
    *,
    retrieve: Callable[[str], list[RetrievedChunk]],
    llm: BaseChatModel,
) -> SparkRagChain:
    """Construye la cadena con un recuperador y un modelo de chat reales."""
    return SparkRagChain(retrieve=retrieve, generator=build_generator(llm))
