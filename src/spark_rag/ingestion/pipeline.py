"""Pipeline de ingesta idempotente del corpus documental."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spark_rag.config import Settings, get_settings
from spark_rag.db.connection import get_connection, get_engine
from spark_rag.db.models import (
    create_run,
    delete_vectors_for_source,
    ensure_vector_index,
    finish_run,
    get_file_record,
    initialize_schema,
    upsert_file_record,
)
from spark_rag.ingestion.cleaner import clean_text
from spark_rag.ingestion.loaders import iter_source_files, load_file
from spark_rag.ingestion.metadata import (
    build_document_metadata,
    compute_sha256,
    make_chunk_id,
    relative_source,
)

logger = logging.getLogger(__name__)

STATUS_INDEXED = "indexed"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class FileResult:
    """Resultado de procesar un archivo del corpus."""

    source: str
    status: str
    sha256: str | None
    chunk_count: int
    error: str | None
    elapsed_seconds: float


@dataclass
class IngestionReport:
    """Resumen de una ejecución completa de ingesta."""

    results: list[FileResult] = field(default_factory=list)

    @property
    def files_discovered(self) -> int:
        return len(self.results)

    @property
    def files_indexed(self) -> int:
        return sum(result.status == STATUS_INDEXED for result in self.results)

    @property
    def files_skipped(self) -> int:
        return sum(result.status == STATUS_SKIPPED for result in self.results)

    @property
    def files_failed(self) -> int:
        return sum(result.status == STATUS_FAILED for result in self.results)

    @property
    def status(self) -> str:
        if not self.results:
            return "empty"
        if self.files_failed == 0:
            return "success"
        if self.files_failed == self.files_discovered:
            return "failed"
        return "partial"

    def summary(self) -> str:
        return (
            f"Descubiertos={self.files_discovered} "
            f"indexados={self.files_indexed} "
            f"omitidos={self.files_skipped} "
            f"fallidos={self.files_failed} "
            f"estado={self.status}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "files_discovered": self.files_discovered,
            "files_indexed": self.files_indexed,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "results": [
                {
                    "source": result.source,
                    "status": result.status,
                    "sha256": result.sha256,
                    "chunk_count": result.chunk_count,
                    "error": result.error,
                    "elapsed_seconds": round(result.elapsed_seconds, 3),
                }
                for result in self.results
            ],
        }


class IngestionPipeline:
    """Orquesta carga, limpieza, chunking, embeddings y control por hash."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        vector_store: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._vector_store = vector_store
        self._embeddings: Any | None = None

    def run(self, data_dir: Path | None = None) -> IngestionReport:
        """Procesa recursivamente el corpus y devuelve el reporte.

        Los errores por archivo se registran y no interrumpen el resto. Un fallo
        de base de datos o de configuración global sí se propaga.
        """
        corpus_dir = (data_dir or self.settings.resolve_data_dir()).resolve()
        files = iter_source_files(corpus_dir)

        with get_connection() as connection:
            initialize_schema(connection)
            run_id = create_run(connection, len(files))

        logger.info(
            "Inicio de ingesta run_id=%s archivos=%d data_dir=%s",
            run_id,
            len(files),
            corpus_dir,
        )

        report = IngestionReport()
        for path in files:
            report.results.append(self._ingest_file(path, corpus_dir))

        with get_connection() as connection:
            finish_run(
                connection,
                run_id=run_id,
                status=report.status,
                files_discovered=report.files_discovered,
                files_indexed=report.files_indexed,
                files_skipped=report.files_skipped,
                files_failed=report.files_failed,
            )

        logger.info("Fin de ingesta run_id=%s %s", run_id, report.summary())
        return report

    def _ingest_file(self, path: Path, data_dir: Path) -> FileResult:
        started = time.monotonic()
        source = relative_source(path, data_dir)
        sha256: str | None = None
        file_size = 0

        try:
            sha256 = compute_sha256(path)
            file_size = path.stat().st_size

            with get_connection() as connection:
                previous = get_file_record(connection, source)

            if (
                previous is not None
                and previous["sha256"] == sha256
                and previous["status"] == STATUS_INDEXED
            ):
                logger.info("Omitido (sin cambios): %s", source)
                return FileResult(
                    source=source,
                    status=STATUS_SKIPPED,
                    sha256=sha256,
                    chunk_count=int(previous["chunk_count"]),
                    error=None,
                    elapsed_seconds=time.monotonic() - started,
                )

            chunk_count = self._index_file(path, data_dir, source)
            with get_connection() as connection:
                upsert_file_record(
                    connection,
                    source=source,
                    sha256=sha256,
                    status=STATUS_INDEXED,
                    file_size=file_size,
                    chunk_count=chunk_count,
                )

            logger.info("Indexado: %s (%d chunks)", source, chunk_count)
            return FileResult(
                source=source,
                status=STATUS_INDEXED,
                sha256=sha256,
                chunk_count=chunk_count,
                error=None,
                elapsed_seconds=time.monotonic() - started,
            )

        except Exception as exc:  # noqa: BLE001 - error por archivo
            error = f"{type(exc).__name__}: {exc}"[:2000]
            logger.error("Error al procesar %s: %s", source, error)
            try:
                with get_connection() as connection:
                    upsert_file_record(
                        connection,
                        source=source,
                        sha256=sha256,
                        status=STATUS_FAILED,
                        file_size=file_size,
                        chunk_count=0,
                        error=error,
                    )
            except Exception as db_exc:  # noqa: BLE001  # pragma: no cover
                logger.error("No se pudo registrar el fallo de %s: %s", source, db_exc)

            return FileResult(
                source=source,
                status=STATUS_FAILED,
                sha256=sha256,
                chunk_count=0,
                error=error,
                elapsed_seconds=time.monotonic() - started,
            )

    def _index_file(self, path: Path, data_dir: Path, source: str) -> int:
        documents = load_file(path)
        chunks = self._build_chunks(path, data_dir, source, documents)
        if not chunks:
            raise ValueError("El documento no contiene texto utilizable.")

        vector_store = self._get_vector_store()
        with get_connection() as connection:
            deleted = delete_vectors_for_source(
                connection,
                collection_name=self.settings.collection_name,
                source=source,
            )
        if deleted:
            logger.info("Eliminados %d chunks previos de %s", deleted, source)

        ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
        vector_store.add_documents(documents=chunks, ids=ids)
        return len(chunks)

    def _build_chunks(
        self,
        path: Path,
        data_dir: Path,
        source: str,
        documents: list[Any],
    ) -> list[Any]:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
            strip_whitespace=True,
        )
        is_pdf = path.suffix.lower() == ".pdf"
        pages: list[Any] = []

        for index, document in enumerate(documents):
            content = clean_text(getattr(document, "page_content", ""))
            if not content:
                continue
            original_metadata = getattr(document, "metadata", {}) or {}
            page = original_metadata.get("page", index if is_pdf else None)
            metadata = build_document_metadata(path, data_dir, page=page)
            pages.append(Document(page_content=content, metadata=metadata))

        chunks = splitter.split_documents(pages) if pages else []
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

        for chunk_index, chunk in enumerate(chunks):
            page = int(chunk.metadata["page"])
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = make_chunk_id(source, page, chunk_index)

        return chunks

    def _get_embeddings(self) -> Any:
        if self._embeddings is None:
            from spark_rag.ingestion.embeddings import get_embeddings

            self._embeddings = get_embeddings()
        return self._embeddings

    def _get_vector_store(self) -> Any:
        if self._vector_store is None:
            from langchain_postgres import PGVector

            self._vector_store = PGVector(
                embeddings=self._get_embeddings(),
                collection_name=self.settings.collection_name,
                connection=get_engine(),
                embedding_length=self.settings.embedding_dim,
                use_jsonb=True,
            )
            with get_connection() as connection:
                ensure_vector_index(connection)
        return self._vector_store
