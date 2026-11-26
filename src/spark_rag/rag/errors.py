"""Excepciones de dominio del motor RAG.

La capa HTTP las traduce a códigos y mensajes controlados; sus textos no deben
incluir credenciales, cadenas de conexión ni trazas.
"""

from __future__ import annotations

import re

_MAX_REASON_CHARS = 200
_SECRET_PATTERNS = (
    # Credenciales en URL (postgresql://user:pass@host).
    (re.compile(r"://[^/\s:@]+:[^/\s@]+@"), "://***@"),
    # Cabeceras y parámetros con credenciales.
    (re.compile(r"(?i)bearer\s+\S+"), "Bearer ***"),
    (re.compile(r"(?i)\b(api[_-]?key|key|token|password)=[^\s&,;]+"), r"\1=***"),
    # Formatos típicos de claves: Google (AIza...), NVIDIA (nvapi-...), OpenAI (sk-...).
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "***"),
    (re.compile(r"nvapi-[0-9A-Za-z_\-]{8,}"), "***"),
    (re.compile(r"\bsk-[0-9A-Za-z_\-]{8,}"), "***"),
)


def describe_error(exc: BaseException) -> str:
    """Motivo breve y seguro de un error, para los logs.

    Incluye el tipo y el comienzo del mensaje (p. ej. «503 This model is
    currently experiencing high demand») y enmascara cualquier cosa con forma de
    credencial. Es una segunda barrera: los secretos no deben llegar al mensaje.
    """
    message = " ".join(str(exc).split())
    for pattern, replacement in _SECRET_PATTERNS:
        message = pattern.sub(replacement, message)
    reason = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    if len(reason) > _MAX_REASON_CHARS:
        reason = reason[: _MAX_REASON_CHARS - 1] + "…"
    return reason


class SparkError(Exception):
    """Base de los errores del motor conversacional."""


class LLMConfigurationError(SparkError):
    """El proveedor o modelo de chat no está configurado correctamente."""


class LLMUnavailable(SparkError):
    """El modelo de chat falló, agotó la cuota o no respondió a tiempo."""


class RetrievalUnavailable(SparkError):
    """No se pudo consultar la memoria documental (pgvector o embeddings)."""


class SessionNotFound(SparkError):
    """El `session_id` recibido no corresponde a ninguna sesión."""


class StorageUnavailable(SparkError):
    """No se pudo leer o guardar el historial de la sesión."""
