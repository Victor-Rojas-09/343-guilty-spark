"""Turnos de conversación y límite del contexto enviado al modelo (MEM-02)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatTurn:
    """Un mensaje del historial de una sesión."""

    role: Role
    content: str


def trim_history(
    turns: Sequence[ChatTurn],
    *,
    max_messages: int,
    max_chars: int,
) -> list[ChatTurn]:
    """Conserva los mensajes más recientes dentro de los límites dados.

    * Como máximo `max_messages` mensajes.
    * La suma de caracteres no supera `max_chars` (se descartan los más
      antiguos; `max_chars=0` desactiva este tope).
    * La ventana no empieza con una respuesta del asistente huérfana.
    """
    if max_messages <= 0:
        return []

    window = list(turns)[-max_messages:]

    if max_chars > 0:
        kept: list[ChatTurn] = []
        total = 0
        for turn in reversed(window):
            total += len(turn.content)
            if total > max_chars:
                break
            kept.append(turn)
        window = list(reversed(kept))

    while window and window[0].role != "user":
        window.pop(0)
    return window
