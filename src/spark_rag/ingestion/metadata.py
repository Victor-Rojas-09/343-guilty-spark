"""Metadatos de procedencia y utilidades de identificación documental.

Los campos mínimos exigidos son `source`, `category` y `page`. También se
deriva `source_type` de la extensión del archivo. Los campos narrativos
opcionales (`work`, `edition`, `level`, `terminal`, `scene`, `speaker`,
`canon_status`, `reliability`) solo se incorporan si existe un sidecar junto al
documento; nunca se inventan.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
SOURCE_TYPE_BY_EXTENSION = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}
OPTIONAL_METADATA_FIELDS = (
    "work",
    "edition",
    "level",
    "terminal",
    "scene",
    "speaker",
    "canon_status",
    "reliability",
)
_CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "343-guilty-spark//chunks")


def compute_sha256(path: Path, *, buffer_size: int = 1024 * 1024) -> str:
    """Calcula el SHA256 de un archivo sin cargarlo completo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(buffer_size), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_source(path: Path, data_dir: Path) -> str:
    """Devuelve la ruta POSIX del archivo relativa al directorio de datos."""
    try:
        relative = path.resolve().relative_to(data_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"El archivo {path} no pertenece a {data_dir}.") from exc
    return PurePosixPath(relative).as_posix()


def infer_category(source: str) -> str:
    """Usa la primera carpeta del corpus como categoría.

    Ejemplo: `history/01_Dossier.pdf` -> `history`. Para archivos en la raíz de
    `data/`, la categoría es `general`.
    """
    parts = PurePosixPath(source).parts
    return parts[0] if len(parts) > 1 else "general"


def infer_source_type(path: Path) -> str:
    """Deriva el tipo de fuente a partir de la extensión del archivo."""
    try:
        return SOURCE_TYPE_BY_EXTENSION[path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"Extensión no soportada: {path.suffix!r}.") from exc


def normalize_page(page: Any, *, is_pdf: bool) -> int:
    """Normaliza el número de página a un entero 1-indexado.

    `PyPDFLoader` expone páginas 0-indexadas. Para Markdown/TXT se usa la
    página lógica 1 cuando no existe un localizador real.
    """
    if page is None:
        return 1
    try:
        number = int(page)
    except (TypeError, ValueError):
        return 1
    if is_pdf:
        return max(1, number + 1)
    return max(1, number)


def _load_sidecar_metadata(path: Path) -> dict[str, Any]:
    """Lee metadatos opcionales desde `<archivo>.metadata.json`.

    Se aceptan las variantes `documento.pdf.metadata.json` y
    `documento.metadata.json`. Solo se incorporan campos permitidos y con
    valores JSON simples.
    """
    candidates = (
        path.with_suffix(path.suffix + ".metadata.json"),
        path.with_suffix(".metadata.json"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        metadata: dict[str, Any] = {}
        for field in OPTIONAL_METADATA_FIELDS:
            value = raw.get(field)
            if value is None or isinstance(value, (dict, list)):
                continue
            metadata[field] = value
        return metadata
    return {}


def build_document_metadata(
    path: Path,
    data_dir: Path,
    *,
    page: Any = None,
) -> dict[str, Any]:
    """Construye los metadatos aplicables a una página o documento.

    Args:
        path: Archivo del corpus.
        data_dir: Raíz de datos para calcular `source` y `category`.
        page: Valor de página original del loader, si existe.
    """
    source = relative_source(path, data_dir)
    source_type = infer_source_type(path)
    metadata: dict[str, Any] = {
        "source": source,
        "category": infer_category(source),
        "page": normalize_page(page, is_pdf=source_type == "pdf"),
        "source_type": source_type,
    }
    metadata.update(_load_sidecar_metadata(path))
    return metadata


def make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """Genera un identificador estable y único para un chunk."""
    name = f"{source}#page={page}#chunk={chunk_index}"
    return str(uuid.uuid5(_CHUNK_NAMESPACE, name))
