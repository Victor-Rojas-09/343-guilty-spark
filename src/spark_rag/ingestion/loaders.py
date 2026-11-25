"""Carga de documentos del corpus con loaders de LangChain.

Se recorren recursivamente `data/` y sus subcarpetas. PDF usa `PyPDFLoader`
(una página por `Document`); Markdown y TXT usan `TextLoader`. El pipeline
captura los errores de cada archivo para no detener el resto del corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spark_rag.ingestion.metadata import SUPPORTED_EXTENSIONS


def iter_source_files(data_dir: Path) -> list[Path]:
    """Devuelve los archivos soportados en orden estable y sin ocultos."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de datos: {data_dir}")

    files: list[Path] = []
    for path in data_dir.rglob("*"):
        relative_parts = path.relative_to(data_dir).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def load_file(path: Path) -> list[Any]:
    """Carga un archivo con el loader de LangChain adecuado."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path)).load()

    if suffix in {".md", ".markdown", ".txt"}:
        from langchain_community.document_loaders import TextLoader

        return TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        ).load()

    raise ValueError(f"Formato no soportado: {path}")
