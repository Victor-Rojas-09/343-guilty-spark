"""Limpieza y normalización del texto extraído del PDF.

Responsabilidad:
    Normalizar espacios y saltos de línea, eliminar encabezados/pies repetidos
    y descartar ruido de maquetación antes del chunking.
"""


def clean_text(text: str) -> str:
    """Normaliza el texto de una página del manual.

    Args:
        text: Texto crudo extraído del PDF.

    Returns:
        Texto limpio, listo para dividir en chunks.

    Raises:
        NotImplementedError: Pendiente de implementación en la Fase 2.
    """
    raise NotImplementedError("Limpieza de texto pendiente de implementación (Fase 2).")
