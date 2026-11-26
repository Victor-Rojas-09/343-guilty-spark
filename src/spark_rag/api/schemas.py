"""Esquemas públicos de la API de conversación (contrato JSON).

El contrato mínimo del manual es `reply`, `mood`, `sources` y `audio_url`.
Se añaden `session_id` (recomendado por el manual) y `source_details` con los
metadatos de procedencia disponibles para que el frontend pueda mostrar
referencias precisas (UI-06).
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from spark_rag.config import get_settings
from spark_rag.rag.mood import Mood


class ChatRequest(BaseModel):
    """Solicitud de `POST /api/chat`."""

    message: str = Field(description="Mensaje del usuario.")
    session_id: UUID | None = Field(
        default=None,
        description="Sesión existente. Si se omite, el backend crea una nueva.",
    )

    @field_validator("message")
    @classmethod
    def _validar_mensaje(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("El mensaje no puede estar vacío.")
        max_chars = get_settings().chat_message_max_chars
        if len(text) > max_chars:
            raise ValueError(f"El mensaje supera el máximo de {max_chars} caracteres.")
        return text


class SourceDetail(BaseModel):
    """Procedencia de un fragmento usado como evidencia.

    Los campos narrativos solo tienen valor cuando existen en los metadatos
    del fragmento; nunca se inventan.
    """

    source: str
    category: str
    page: int | None = None
    source_class: str = Field(
        description="Clase de autoridad: canon, transcripciones, referencia, "
        "comunidad o sin_clasificar."
    )
    similarity: float
    work: str | None = None
    edition: str | None = None
    level: str | None = None
    terminal: str | None = None
    scene: str | None = None
    speaker: str | None = None


class ChatResponse(BaseModel):
    """Respuesta de `POST /api/chat`."""

    reply: str
    mood: Mood
    sources: list[str] = Field(default_factory=list)
    audio_url: str | None = None
    session_id: UUID
    source_details: list[SourceDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Cuerpo de error controlado: sin trazas, rutas ni credenciales."""

    error: str
    detail: str
