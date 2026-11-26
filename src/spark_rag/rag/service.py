"""Servicio de conversación: sesiones + cadena RAG + persistencia.

Orden de un turno:

1. Resolver la sesión (nueva si no llega `session_id`; 404 si no existe).
2. Cargar el historial de esa sesión y recortarlo (MEM-02).
3. Ejecutar la cadena RAG con una consulta de recuperación que incluye el
   último turno del usuario, para que las preguntas de seguimiento recuperen
   el contexto pertinente sin una llamada extra al modelo.
4. Guardar usuario + asistente en una única transacción, solo si la
   generación tuvo éxito: un fallo de Gemini no deja turnos a medias.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol
from uuid import UUID, uuid4

from spark_rag.config import Settings, get_settings
from spark_rag.rag.chain import RagResult
from spark_rag.rag.errors import SessionNotFound, StorageUnavailable
from spark_rag.rag.history import ChatTurn, trim_history
from spark_rag.rag.mood import Mood
from spark_rag.rag.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)


class SessionStore(Protocol):
    """Almacenamiento de sesiones que necesita el servicio."""

    def exists(self, session_id: UUID) -> bool: ...

    def recent_messages(self, session_id: UUID, limit: int) -> list[dict[str, Any]]: ...

    def save_exchange(self, **kwargs: Any) -> None: ...


class RagChain(Protocol):
    """Interfaz mínima de la cadena RAG."""

    def invoke(
        self,
        question: str,
        *,
        history: Sequence[ChatTurn] = (),
        retrieval_query: str | None = None,
    ) -> RagResult: ...


@dataclass(frozen=True)
class ChatReply:
    """Resultado de un turno listo para serializar en la API."""

    reply: str
    mood: Mood
    session_id: UUID
    sources: list[str] = field(default_factory=list)
    source_details: list[dict[str, Any]] = field(default_factory=list)
    audio_url: str | None = None


def build_retrieval_query(message: str, history: Sequence[ChatTurn]) -> str:
    """Mensaje actual precedido del último mensaje del usuario, si existe."""
    for turn in reversed(history):
        if turn.role == "user" and turn.content.strip():
            return f"{turn.content.strip()}\n{message.strip()}"
    return message.strip()


def _to_turns(rows: Sequence[dict[str, Any]]) -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    for row in rows:
        role = row.get("role")
        content = str(row.get("content") or "")
        if role in ("user", "assistant") and content:
            turns.append(ChatTurn(role=role, content=content))
    return turns


class ChatService:
    """Coordina sesión, historial, cadena RAG y persistencia."""

    def __init__(
        self,
        *,
        chain: RagChain,
        store: SessionStore,
        settings: Settings | None = None,
        model_name: str = "",
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self._chain = chain
        self._store = store
        self._settings = settings or get_settings()
        self._model_name = model_name
        self._prompt_version = prompt_version

    def _load_history(self, session_id: UUID) -> list[ChatTurn]:
        try:
            if not self._store.exists(session_id):
                raise SessionNotFound("La sesión indicada no existe.")
            rows = self._store.recent_messages(
                session_id, self._settings.chat_history_max_messages
            )
        except SessionNotFound:
            raise
        except Exception as exc:  # se traduce a error de dominio
            logger.error("No se pudo leer la sesión: %s", type(exc).__name__)
            raise StorageUnavailable("El historial no está disponible.") from exc

        return trim_history(
            _to_turns(rows),
            max_messages=self._settings.chat_history_max_messages,
            max_chars=self._settings.chat_history_max_chars,
        )

    def chat(self, message: str, session_id: UUID | None = None) -> ChatReply:
        """Procesa un mensaje del usuario y devuelve la respuesta de Spark."""
        if session_id is None:
            session_id = uuid4()
            history: list[ChatTurn] = []
        else:
            history = self._load_history(session_id)

        result = self._chain.invoke(
            message,
            history=history,
            retrieval_query=build_retrieval_query(message, history),
        )
        details = result.source_details

        try:
            self._store.save_exchange(
                session_id=session_id,
                user_message=message,
                assistant_message=result.reply,
                mood=result.mood.value,
                sources=details,
                prompt_version=self._prompt_version,
                model=self._model_name,
            )
        except Exception as exc:  # se traduce a error de dominio
            logger.error("No se pudo guardar el turno: %s", type(exc).__name__)
            raise StorageUnavailable("No se pudo guardar la conversación.") from exc

        logger.info(
            "Turno completado session=%s historial=%d fragmentos=%d mood=%s",
            session_id,
            len(history),
            len(result.chunks),
            result.mood.value,
        )
        return ChatReply(
            reply=result.reply,
            mood=result.mood,
            session_id=session_id,
            sources=result.sources,
            source_details=details,
        )


def build_chat_service(settings: Settings | None = None) -> ChatService:
    """Construye el servicio con PostgreSQL, pgvector y el modelo configurado.

    Raises:
        LLMConfigurationError: Si el proveedor de chat no está configurado.
    """
    from spark_rag.db.sessions import PostgresSessionStore
    from spark_rag.rag.chain import SparkRagChain
    from spark_rag.rag.fallback import build_fallback_generator
    from spark_rag.rag.llm import chat_model_name
    from spark_rag.rag.retriever import build_retriever

    settings = settings or get_settings()
    # Principal y respaldos (`LLM_FALLBACK_PROVIDERS`), antes que el recuperador.
    generator = build_fallback_generator(settings)
    retriever = build_retriever(settings)
    return ChatService(
        chain=SparkRagChain(retrieve=retriever.retrieve, generator=generator),
        store=PostgresSessionStore(),
        settings=settings,
        model_name=chat_model_name(settings),
    )


@lru_cache
def get_chat_service() -> ChatService:
    """Instancia reutilizable del servicio de chat."""
    return build_chat_service()
