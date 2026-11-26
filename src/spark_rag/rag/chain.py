"""Cadena RAG de LangChain (RAG-01).

Flujo: consulta → embedding + recuperación en pgvector → contexto delimitado
ordenado por autoridad → System Prompt + historial → modelo de chat con salida
estructurada `{reply, mood}`.

La salida estructurada se pide con `include_raw=True`: si el modelo no respeta
el esquema, se recupera el texto crudo como `reply` y `mood` cae a `thinking`
en lugar de fallar la petición. El esquema, el parser y los modos del
proveedor compatible con OpenAI están en `rag/structured.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from spark_rag.rag.errors import (
    LLMUnavailable,
    RetrievalUnavailable,
    describe_error,
)
from spark_rag.rag.history import ChatTurn
from spark_rag.rag.mood import Mood
from spark_rag.rag.prompts import build_messages
from spark_rag.rag.sources import (
    RetrievedChunk,
    format_context,
    public_sources,
    source_details,
)

# Reexportados: el esquema y el parser viven en `structured`.
from spark_rag.rag.structured import SparkOutput, parse_generation

__all__ = [
    "RagResult",
    "SparkOutput",
    "SparkRagChain",
    "build_chain",
    "build_generator",
    "parse_generation",
]

logger = logging.getLogger(__name__)


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
