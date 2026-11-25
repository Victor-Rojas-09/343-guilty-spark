"""Limpieza y normalización del texto extraído de PDF, Markdown o TXT."""

from __future__ import annotations

import re
import unicodedata

_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\f\v]+")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_HYPHENATED_LINE_BREAK = re.compile(r"(?<=[A-Za-zÁÉÍÓÚÜÑáéíóúüñ])-\n(?=[a-záéíóúüñ])")


def clean_text(text: str) -> str:
    """Normaliza un fragmento de texto antes del chunking.

    - Unifica saltos de línea y espacios.
    - Elimina caracteres de control y espacios de ancho cero.
    - Recompone palabras cortadas por guion al final de línea.
    - Reduce líneas en blanco consecutivas sin perder la estructura de párrafos.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = "".join(
        char
        for char in text
        if char in {"\n", "\t"} or unicodedata.category(char)[0] != "C"
    )
    text = _HYPHENATED_LINE_BREAK.sub("", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _HORIZONTAL_WHITESPACE.sub(" ", text)
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)
    return text.strip()
