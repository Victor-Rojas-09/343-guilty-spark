"""Estados de ánimo (`mood`) controlados del personaje.

El conjunto es el contrato común entre backend y frontend (API-03, AV-02):
el modelo puede proponer un valor, pero nunca se devuelve texto arbitrario.
"""

from __future__ import annotations

from enum import StrEnum


class Mood(StrEnum):
    """Estados visuales admitidos para el monitor/orbe."""

    IDLE = "idle"
    THINKING = "thinking"
    ENTHUSIASTIC = "enthusiastic"
    RAMPANT = "rampant"


DEFAULT_MOOD = Mood.THINKING


def normalize_mood(value: object) -> Mood:
    """Convierte un valor propuesto por el modelo en un `Mood` válido.

    Acepta instancias de `Mood` o cadenas con mayúsculas/espacios. Cualquier
    otro valor se sustituye por `DEFAULT_MOOD`.
    """
    if isinstance(value, Mood):
        return value
    if isinstance(value, str):
        try:
            return Mood(value.strip().lower())
        except ValueError:
            return DEFAULT_MOOD
    return DEFAULT_MOOD
