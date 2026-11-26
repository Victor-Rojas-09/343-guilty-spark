"""Fábrica del modelo de chat (RAG-02).

El resto del sistema depende solo de `BaseChatModel` de LangChain; el proveedor
y el modelo se eligen por configuración (`LLM_PROVIDER`, `GEMINI_CHAT_MODEL`).
Para añadir otro proveedor basta con registrar un constructor en
`_PROVIDERS`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from spark_rag.config import Settings, get_settings
from spark_rag.rag.errors import LLMConfigurationError


def _build_gemini(settings: Settings) -> BaseChatModel:
    if not settings.gemini_api_key:
        raise LLMConfigurationError(
            "GEMINI_API_KEY no está configurada. Añádela a .env o al entorno."
        )
    if not settings.gemini_chat_model.strip():
        raise LLMConfigurationError("GEMINI_CHAT_MODEL no puede estar vacío.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model.strip(),
        google_api_key=settings.gemini_api_key,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


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
