"""Aplicación FastAPI.

Expone `GET /health` (PostgreSQL + pgvector) y `POST /api/chat` (Etapa 2).
Los errores se devuelven siempre con el cuerpo controlado `ErrorResponse`, sin
trazas internas, rutas, cadenas de conexión ni credenciales.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from spark_rag import __version__
from spark_rag.api.chat import router as chat_router
from spark_rag.db.connection import check_database
from spark_rag.rag.errors import (
    LLMConfigurationError,
    LLMUnavailable,
    RetrievalUnavailable,
    SessionNotFound,
    SparkError,
    StorageUnavailable,
)

logger = logging.getLogger(__name__)

# (código HTTP, identificador, mensaje público) por tipo de error de dominio.
_ERROR_MAP: dict[type[SparkError], tuple[int, str, str]] = {
    SessionNotFound: (
        404,
        "session_not_found",
        "La sesión indicada no existe. Inicia una conversación sin session_id.",
    ),
    LLMConfigurationError: (
        503,
        "llm_not_configured",
        "El modelo de conversación no está configurado.",
    ),
    LLMUnavailable: (
        503,
        "llm_unavailable",
        "El modelo de conversación no está disponible. Inténtalo más tarde.",
    ),
    RetrievalUnavailable: (
        503,
        "retrieval_unavailable",
        "La memoria documental no está disponible. Inténtalo más tarde.",
    ),
    StorageUnavailable: (
        503,
        "storage_unavailable",
        "El historial de conversación no está disponible. Inténtalo más tarde.",
    ),
}


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": error, "detail": detail}
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Crea el esquema de sesiones sin impedir el arranque si la BD no responde."""
    try:
        from spark_rag.db.sessions import PostgresSessionStore

        PostgresSessionStore().initialize()
    except Exception as exc:  # noqa: BLE001 - /health informará del estado
        logger.warning("No se pudo preparar el esquema de sesiones: %s", exc)
    yield


app = FastAPI(
    title="343 Guilty Spark // RAG",
    version=__version__,
    description="Asistente conversacional con RAG: salud, chat y sesiones (Etapa 2).",
    lifespan=lifespan,
)
app.include_router(chat_router)


@app.exception_handler(SparkError)
async def spark_error_handler(_: Request, exc: SparkError) -> JSONResponse:
    status_code, error, detail = _ERROR_MAP.get(
        type(exc), (503, "service_unavailable", "Servicio no disponible.")
    )
    return _error(status_code, error, detail)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    problems: list[str] = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
        message = str(item.get("msg", "valor inválido")).removeprefix("Value error, ")
        problems.append(f"{location}: {message}" if location else message)
    detail = "; ".join(problems) or "Solicitud inválida."
    return _error(422, "invalid_request", detail)


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Error no controlado: %s", type(exc).__name__)
    return _error(500, "internal_error", "Error interno del servidor.")


@app.get("/health")
def health() -> JSONResponse:
    """Comprueba base de datos y pgvector."""
    status = check_database()
    payload = {
        "status": "ok" if status.ok else "degraded",
        "service": "spark-rag",
        "version": __version__,
        "database": status.error is None or status.server_version is not None,
        "pgvector": status.pgvector_version is not None,
        "pgvector_version": status.pgvector_version,
    }
    if status.error and status.pgvector_version is None:
        payload["detail"] = status.error
    return JSONResponse(
        content=payload,
        status_code=200 if status.ok else 503,
    )
