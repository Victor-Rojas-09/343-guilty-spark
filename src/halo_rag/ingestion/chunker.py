"""División del texto en chunks con metadatos para citar la fuente.

Responsabilidad:
    Aplicar una estrategia de chunking (tamaño/solapamiento configurables) y
    asociar a cada chunk el documento, la página y su índice.
"""

from dataclasses import dataclass, field
from typing import Any

from halo_rag.ingestion.pdf_reader import PageText


@dataclass(frozen=True)
class Chunk:
    """Fragmento de texto listo para generar su embedding."""

    content: str
    page_number: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_pages(pages: list[PageText], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Divide las páginas limpias en chunks solapados.

    Args:
        pages: Páginas extraídas del PDF.
        chunk_size: Tamaño objetivo del chunk.
        chunk_overlap: Solapamiento entre chunks consecutivos.

    Returns:
        Lista de chunks con metadatos de documento y página.

    Raises:
        NotImplementedError: Pendiente de implementación en la Fase 2.
    """
    raise NotImplementedError("Chunking pendiente de implementación (Fase 2).")
