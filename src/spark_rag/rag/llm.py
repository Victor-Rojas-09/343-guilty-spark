"""Fábrica del modelo de chat (RAG-02).

El resto del sistema depende solo de `BaseChatModel` de LangChain; el proveedor
y el modelo se eligen por configuración (`LLM_PROVIDER`, `GEMINI_CHAT_MODEL`,
`OPENAI_COMPAT_*`). Para añadir otro proveedor basta con registrar un
constructor en `_PROVIDERS`.

Proveedores:
- `gemini`: `ChatGoogleGenerativeAI` (langchain-google-genai).
- `openai_compat`: cualquier API compatible con OpenAI (p. ej. NVIDIA) vía
  `BaseChatOpenAI` de langchain-openai. Se usa la clase base y no `ChatOpenAI`
  porque esta renombra `max_tokens` a `max_completion_tokens` y aplica ajustes
  propios de OpenAI que los servidores compatibles no siempre aceptan.

Las dependencias de cada proveedor se importan solo al construirlo.

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
        timeout=settings.chat_timeout_seconds("gemini"),
        max_retries=settings.llm_max_retries,
    )
    disable_internal_retry(model)
    return model


def _build_openai_compat(settings: Settings) -> BaseChatModel:
    api_key = settings.openai_compat_api_key.get_secret_value().strip()
    if not api_key:
        raise LLMConfigurationError(
            "OPENAI_COMPAT_API_KEY no está configurada. Añádela a .env o al entorno."
        )
    model_name = settings.openai_compat_chat_model.strip()
    if not model_name:
        raise LLMConfigurationError("OPENAI_COMPAT_CHAT_MODEL no puede estar vacío.")

    try:
        from langchain_openai.chat_models.base import BaseChatOpenAI
    except ImportError as exc:
        raise LLMConfigurationError(
            "El proveedor openai_compat necesita langchain-openai: "
            "instala requirements.txt o reconstruye la imagen."
        ) from exc

    return BaseChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=settings.openai_compat_base_url,
        temperature=settings.openai_compat_effective_temperature,
        top_p=settings.openai_compat_top_p,
        max_tokens=settings.openai_compat_max_output_tokens,
        timeout=settings.chat_timeout_seconds("openai_compat"),
        max_retries=settings.llm_max_retries,
        extra_body=settings.openai_compat_extra_body_dict or None,
    )


_PROVIDERS: dict[str, Callable[[Settings], BaseChatModel]] = {
    "gemini": _build_gemini,
    "openai_compat": _build_openai_compat,
}


def supported_providers() -> tuple[str, ...]:
    """Proveedores de chat disponibles."""
    return tuple(sorted(_PROVIDERS))


def build_chat_model(
    settings: Settings | None = None, provider: str | None = None
) -> BaseChatModel:
    """Construye el modelo de chat de `provider` (por defecto, `LLM_PROVIDER`).

    Raises:
        LLMConfigurationError: Proveedor desconocido o credenciales ausentes.
    """
    settings = settings or get_settings()
    name = (provider or settings.llm_provider).strip().lower()
    builder = _PROVIDERS.get(name)
    if builder is None:
        raise LLMConfigurationError(
            f"LLM_PROVIDER={name!r} no está soportado. "
            f"Opciones: {', '.join(supported_providers())}."
        )
    return builder(settings)


def chat_model_name(
    settings: Settings | None = None, provider: str | None = None
) -> str:
    """Identificador legible del modelo (para persistencia y evaluaciones).

    Gemini conserva el nombre del modelo a secas (compatibilidad con los
    registros anteriores); el proveedor compatible usa `etiqueta:modelo`.
    """
    settings = settings or get_settings()
    name = (provider or settings.llm_provider).strip().lower()
    if name == "gemini":
        return settings.gemini_chat_model
    if name == "openai_compat":
        return f"{settings.openai_compat_label}:{settings.openai_compat_chat_model}"
    return name


def generation_settings(
    settings: Settings | None = None, provider: str | None = None
) -> dict[str, object]:
    """Parámetros de generación de un proveedor, para informes y registros.

    No incluye credenciales. `structured_mode` es `native` en Gemini
    (`with_structured_output`) y el modo configurado en `openai_compat`.
    """
    settings = settings or get_settings()
    name = (provider or settings.llm_provider).strip().lower()
    if name == "openai_compat":
        return {
            "provider": name,
            "model": chat_model_name(settings, name),
            "temperature": settings.openai_compat_effective_temperature,
            "top_p": settings.openai_compat_top_p,
            "structured_mode": settings.openai_compat_structured_mode,
            "max_output_tokens": settings.openai_compat_max_output_tokens,
            "timeout_seconds": settings.chat_timeout_seconds(name),
        }
    return {
        "provider": name,
        "model": chat_model_name(settings, name),
        "temperature": settings.llm_temperature,
        "top_p": None,
        "structured_mode": "native",
        "max_output_tokens": settings.llm_max_output_tokens,
        "timeout_seconds": settings.chat_timeout_seconds(name),
    }


@lru_cache
def get_chat_model() -> BaseChatModel:
    """Instancia reutilizable del modelo de chat configurado."""
    return build_chat_model()
