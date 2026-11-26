"""Turnos de conversación independientes del almacenamiento."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatTurn:
    """Un mensaje del historial de una sesión."""

    role: Role
    content: str
