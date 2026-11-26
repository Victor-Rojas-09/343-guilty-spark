"""Clasificación de errores del modelo de chat para la cadena de respaldo.

Solo los fallos de disponibilidad pasan al siguiente proveedor: cuota o límite
por minuto (429), servidor saturado o caído (5xx), timeout, error de conexión y
respuesta vacía. Los errores de configuración (400/401/403/404: clave
incorrecta, modelo inexistente, petición inválida) NO activan el respaldo, para
no ocultar un `.env` mal configurado.

Se reconocen sin importar los SDK: por el código HTTP (`code` en
`google.api_core`, `status_code` en `openai`) y por el nombre de la clase de la
excepción, así el módulo no depende de que `openai` esté instalado.
"""

from __future__ import annotations

from enum import StrEnum

from spark_rag.rag.errors import EmptyModelResponse


class ErrorKind(StrEnum):
    """Tipo de fallo de una llamada al modelo de chat."""

    RATE_LIMIT = "rate_limit"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    EMPTY_RESPONSE = "empty_response"
    CONFIGURATION = "configuration"
    OTHER = "other"


FALLBACK_KINDS = frozenset(
    {
        ErrorKind.RATE_LIMIT,
        ErrorKind.UNAVAILABLE,
        ErrorKind.TIMEOUT,
        ErrorKind.CONNECTION,
        ErrorKind.EMPTY_RESPONSE,
    }
)

_UNAVAILABLE_STATUS = frozenset({500, 502, 503, 504})
# `DeadlineExceeded` (google.api_core), `APITimeoutError` (openai) y
# `TimeoutException` (httpx, base de ReadTimeout/ConnectTimeout...).
_TIMEOUT_NAMES = frozenset({"DeadlineExceeded", "APITimeoutError", "TimeoutException"})
# `APIConnectionError` (openai) y `TransportError` (httpx, base de ConnectError).
_CONNECTION_NAMES = frozenset({"APIConnectionError", "TransportError"})
_MAX_CHAIN = 5


def _status_code(exc: BaseException) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 100 <= value < 600
        ):
            return value
    return None


def _classify_one(exc: BaseException) -> ErrorKind:
    if isinstance(exc, EmptyModelResponse):
        return ErrorKind.EMPTY_RESPONSE

    names = {cls.__name__ for cls in type(exc).__mro__}
    # El timeout se mira antes que la conexión: en openai, APITimeoutError
    # hereda de APIConnectionError.
    if names & _TIMEOUT_NAMES or isinstance(exc, TimeoutError):
        return ErrorKind.TIMEOUT

    status = _status_code(exc)
    if status == 429:
        return ErrorKind.RATE_LIMIT
    if status == 408:
        return ErrorKind.TIMEOUT
    if status in _UNAVAILABLE_STATUS:
        return ErrorKind.UNAVAILABLE
    if status is not None and 400 <= status < 500:
        return ErrorKind.CONFIGURATION

    if names & _CONNECTION_NAMES or isinstance(exc, ConnectionError):
        return ErrorKind.CONNECTION
    return ErrorKind.OTHER


def classify_error(exc: BaseException) -> ErrorKind:
    """Clasifica un error del modelo, mirando también sus causas encadenadas."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(seen) < _MAX_CHAIN:
        seen.add(id(current))
        kind = _classify_one(current)
        if kind is not ErrorKind.OTHER:
            return kind
        current = current.__cause__ or current.__context__
    return ErrorKind.OTHER


def should_fallback(exc: BaseException) -> bool:
    """Si el error debe pasar al siguiente proveedor de la cadena."""
    return classify_error(exc) in FALLBACK_KINDS
