"""Lectura del PDF del manual de Halo: Combat Evolved.

Responsabilidad:
    Extraer el texto de cada página conservando el número de página como
    metadato, necesario para citar la fuente en las respuestas.

Nota: la implementación concreta (pypdf/pdfplumber) corresponde a la Fase 2.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """Texto extraído de una página del manual."""

    page_number: int  # 1-indexado
    text: str


def read_pages(pdf_path: str) -> list[PageText]:
    """Extrae el texto de cada página del PDF indicado.

    Args:
        pdf_path: Ruta al PDF del manual.

    Returns:
        Lista de páginas con su número y texto.

    Raises:
        NotImplementedError: Pendiente de implementación en la Fase 2.
    """
    raise NotImplementedError("Lectura de PDF pendiente de implementación (Fase 2).")
