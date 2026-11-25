"""Aplicación FastAPI de la Etapa 1.

Solo expone `GET /health`, que verifica la conexión a PostgreSQL y que la
extensión `vector` esté habilitada. El chat RAG se incorporará en la Etapa 2.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from spark_rag import __version__
from spark_rag.db.connection import check_database

app = FastAPI(
    title="343 Guilty Spark // RAG",
    version=__version__,
    description="Infraestructura e ingesta documental (Etapa 1).",
)


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
