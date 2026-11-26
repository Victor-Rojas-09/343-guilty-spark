import httpx
import pytest
from google.api_core import exceptions as google_exceptions

from spark_rag.rag.errors import EmptyModelResponse, LLMUnavailable
from spark_rag.rag.llm_errors import ErrorKind, classify_error, should_fallback


def _openai_like(name: str, status_code: int | None = None, base: type = Exception):
    """Excepción con el nombre y el `status_code` de las de `openai`, sin importarlo."""
    attributes = {} if status_code is None else {"status_code": status_code}
    return type(name, (base,), attributes)("mensaje")


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        # Gemini (google.api_core), tal y como llegan desde langchain-google-genai.
        (google_exceptions.ResourceExhausted("quota"), ErrorKind.RATE_LIMIT),
        (google_exceptions.ServiceUnavailable("high demand"), ErrorKind.UNAVAILABLE),
        (google_exceptions.InternalServerError("x"), ErrorKind.UNAVAILABLE),
        (google_exceptions.BadGateway("x"), ErrorKind.UNAVAILABLE),
        (google_exceptions.GatewayTimeout("x"), ErrorKind.UNAVAILABLE),
        (google_exceptions.DeadlineExceeded("x"), ErrorKind.TIMEOUT),
        # Proveedor compatible con OpenAI (mismos nombres y códigos que `openai`).
        (_openai_like("RateLimitError", 429), ErrorKind.RATE_LIMIT),
        (_openai_like("InternalServerError", 500), ErrorKind.UNAVAILABLE),
        (_openai_like("APIStatusError", 503), ErrorKind.UNAVAILABLE),
        (_openai_like("APIStatusError", 502), ErrorKind.UNAVAILABLE),
        (_openai_like("APITimeoutError"), ErrorKind.TIMEOUT),
        (_openai_like("APIConnectionError"), ErrorKind.CONNECTION),
        (_openai_like("APIStatusError", 408), ErrorKind.TIMEOUT),
        # httpx y excepciones estándar.
        (httpx.ReadTimeout("x"), ErrorKind.TIMEOUT),
        (httpx.ConnectTimeout("x"), ErrorKind.TIMEOUT),
        (httpx.ConnectError("x"), ErrorKind.CONNECTION),
        (TimeoutError(), ErrorKind.TIMEOUT),
        (ConnectionResetError(), ErrorKind.CONNECTION),
        # Respuesta vacía tras limpiar (p. ej. razonamiento que agotó los tokens).
        (EmptyModelResponse("vacía"), ErrorKind.EMPTY_RESPONSE),
    ],
)
def test_fallos_de_disponibilidad_activan_el_respaldo(
    error: BaseException, kind: ErrorKind
) -> None:
    assert classify_error(error) is kind
    assert should_fallback(error) is True


@pytest.mark.parametrize(
    "error",
    [
        google_exceptions.Unauthenticated("API key not valid"),
        google_exceptions.PermissionDenied("x"),
        google_exceptions.NotFound("models/gemini-x is not found"),
        google_exceptions.InvalidArgument("x"),
        _openai_like("AuthenticationError", 401),
        _openai_like("PermissionDeniedError", 403),
        _openai_like("NotFoundError", 404),
        _openai_like("BadRequestError", 400),
        _openai_like("UnprocessableEntityError", 422),
    ],
)
def test_errores_de_configuracion_no_activan_el_respaldo(error: BaseException) -> None:
    assert classify_error(error) is ErrorKind.CONFIGURATION
    assert should_fallback(error) is False


@pytest.mark.parametrize("error", [ValueError("x"), RuntimeError("x"), KeyError("x")])
def test_errores_desconocidos_no_activan_el_respaldo(error: BaseException) -> None:
    assert classify_error(error) is ErrorKind.OTHER
    assert should_fallback(error) is False


def _raise_chained(cause: BaseException) -> LLMUnavailable:
    try:
        raise LLMUnavailable("El modelo de chat no respondió.") from cause
    except LLMUnavailable as exc:
        return exc


def test_mira_la_causa_encadenada() -> None:
    assert should_fallback(_raise_chained(google_exceptions.ServiceUnavailable("x")))
    assert not should_fallback(_raise_chained(google_exceptions.Unauthenticated("x")))


def test_ignora_codigos_que_no_son_http() -> None:
    error = _openai_like("RareError")
    error.code = "UNAVAILABLE"
    error.status_code = True
    assert classify_error(error) is ErrorKind.OTHER


def test_cadena_circular_no_se_cuelga() -> None:
    first, second = ValueError("a"), ValueError("b")
    first.__cause__, second.__cause__ = second, first
    assert classify_error(first) is ErrorKind.OTHER
