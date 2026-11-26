"""Endpoint de conversación `POST /api/chat` (API-01, API-02)."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from spark_rag.api.schemas import ChatRequest, ChatResponse, ErrorResponse, SourceDetail
from spark_rag.rag.errors import LLMConfigurationError, RetrievalUnavailable
from spark_rag.rag.service import ChatService, get_chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def get_service() -> ChatService:
    """Dependencia sustituible en pruebas mediante `app.dependency_overrides`."""
    try:
        return get_chat_service()
    except LLMConfigurationError:
        raise
    except Exception as exc:  # se traduce a error de dominio
        logger.error("No se pudo inicializar el servicio: %s", type(exc).__name__)
        raise RetrievalUnavailable("La memoria documental no está disponible.") from exc


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Sesión inexistente."},
        422: {"model": ErrorResponse, "description": "Solicitud inválida."},
        503: {"model": ErrorResponse, "description": "Dependencia no disponible."},
    },
)
def chat(
    request: ChatRequest, service: Annotated[ChatService, Depends(get_service)]
) -> ChatResponse:
    """Responde en personaje usando la memoria documental y la sesión indicada."""
    result = service.chat(request.message, request.session_id)
    return ChatResponse(
        reply=result.reply,
        mood=result.mood,
        sources=result.sources,
        audio_url=None,
        session_id=result.session_id,
        source_details=[
            SourceDetail.model_validate(detail) for detail in result.source_details
        ],
    )
