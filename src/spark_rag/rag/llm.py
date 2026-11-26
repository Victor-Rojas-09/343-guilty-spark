"""Fábrica del modelo de chat (RAG-02).

El resto del sistema depende solo de `BaseChatModel` de LangChain; el proveedor
y el modelo se eligen por configuración (`LLM_PROVIDER`, `GEMINI_CHAT_MODEL`).
Para añadir otro proveedor basta con registrar un constructor en
`_PROVIDERS`.

`LLM_MAX_RETRIES` debe ser la única política de reintentos: el cliente gapic que
usa `langchain-google-genai` 2.x reintenta por su cuenta los 503 (hasta 600 s)
y convierte una llamada en muchas peticiones, lo que agota el límite por minuto.
Por eso se desactiva ese reintento interno tras construir el modelo.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from spark_rag.config import Settings, get_settings
from spark_rag.rag.errors import LLMConfigurationError

logger = logging.getLogger(__name__)


def disable_internal_retry(model: Any) -> bool:
    """Quita el reintento interno del cliente gapic en `generate_content`.

    Depende de atributos privados de `google-ai-generativelanguage`; si cambian,
    solo se emite un aviso y el modelo sigue funcionando (con el reintento
    interno de la librería). Devuelve si se pudo desactivar.
    """
    try:
        transport = model.client._transport
        wrapped = transport._wrapped_methods[transport.generate_content]
        if not hasattr(wrapped, "_retry"):
            raise AttributeError("_retry")
        wrapped._retry = None
    except Exception as exc:  # noqa: BLE001 - atributos privados: nunca rompe la app
        logger.warning(
            "No se pudo desactivar el reintento interno de Gemini (%s); "
            "LLM_MAX_RETRIES puede no ser la única política de reintentos.",
            type(exc).__name__,
        )
        return False
    return True


def _build_gemini(settings: Settings) -> BaseChatModel:
    if not settings.gemini_api_key:
        raise LLMConfigurationError(
            "GEMINI_API_KEY no está configurada. Añádela a .env o al entorno."
        )
    if not settings.gemini_chat_model.strip():
        raise LLMConfigurationError("GEMINI_CHAT_MODEL no puede estar vacío.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model.strip(),
        google_api_key=settings.gemini_api_key,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    disable_internal_retry(model)
    return model


_PROVIDERS: dict[str, Callable[[Settings], BaseChatModel]] = {
    "gemini": _build_gemini,
}


def supported_providers() -> tuple[str, ...]:
    """Proveedores de chat disponibles."""
    return tuple(sorted(_PROVIDERS))


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Construye el modelo de chat configurado.

    Raises:
        LLMConfigurationError: Proveedor desconocido o credenciales ausentes.
    """
    settings = settings or get_settings()
    builder = _PROVIDERS.get(settings.llm_provider)
    if builder is None:
        raise LLMConfigurationError(
            f"LLM_PROVIDER={settings.llm_provider!r} no está soportado. "
            f"Opciones: {', '.join(supported_providers())}."
        )
    return builder(settings)


def chat_model_name(settings: Settings | None = None) -> str:
    """Identificador legible del modelo (para persistencia y evaluaciones)."""
    settings = settings or get_settings()
    if settings.llm_provider == "gemini":
        return settings.gemini_chat_model
    return settings.llm_provider


@lru_cache
def get_chat_model() -> BaseChatModel:
    """Instancia reutilizable del modelo de chat configurado."""
    return build_chat_model()
