"""Cadena de respaldo del modelo de chat (`LLM_FALLBACK_PROVIDERS`).

`FallbackGenerator` prueba en orden el proveedor principal (`LLM_PROVIDER`) y
los de respaldo. Pasa al siguiente solo ante fallos de disponibilidad (429,
5xx, timeout, conexión o respuesta vacía; ver `llm_errors.should_fallback`).
Un error de configuración (400/401/403/404) corta la cadena: no se oculta un
`.env` mal configurado.

Cada intento queda registrado como `Attempt(model, outcome, seconds)`, tanto en
el resultado como en el `LLMUnavailable` final, para el informe de evaluación.
Sin respaldo configurado hay un único candidato y el comportamiento es el de
siempre: cualquier fallo es `LLMUnavailable`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from spark_rag.config import Settings, get_settings
from spark_rag.rag.errors import EmptyModelResponse, LLMUnavailable, describe_error
from spark_rag.rag.llm import build_chat_model, chat_model_name, generation_settings
from spark_rag.rag.llm_errors import FALLBACK_KINDS, classify_error
from spark_rag.rag.mood import Mood
from spark_rag.rag.structured import (
    build_native_generator,
    build_structured_generator,
    parse_generation,
)

logger = logging.getLogger(__name__)

OUTCOME_OK = "ok"


@dataclass(frozen=True)
class Attempt:
    """Un intento de generación: modelo, resultado (`ok` o tipo de error) y duración."""

    model: str
    outcome: str
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "outcome": self.outcome,
            "seconds": round(self.seconds, 3),
        }


@dataclass(frozen=True)
class Candidate:
    """Un proveedor de la cadena con su generador `{reply, mood}` ya construido."""

    model: str
    generator: Runnable[list[BaseMessage], Any]
    structured_mode: str = ""


@dataclass(frozen=True)
class Generation:
    """Respuesta normalizada del primer proveedor que respondió."""

    reply: str
    mood: Mood
    structured: bool
    model: str
    structured_mode: str
    attempts: tuple[Attempt, ...]


class FallbackGenerator:
    """Prueba los candidatos en orden hasta obtener una respuesta útil."""

    def __init__(
        self,
        candidates: Sequence[Candidate],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not candidates:
            raise ValueError("La cadena de chat necesita al menos un proveedor.")
        self._candidates = tuple(candidates)
        self._clock = clock

    @property
    def models(self) -> list[str]:
        return [candidate.model for candidate in self._candidates]

    def generate(self, messages: list[BaseMessage]) -> Generation:
        """Genera la respuesta; `LLMUnavailable` (con los intentos) si nadie responde."""
        attempts: list[Attempt] = []
        last = len(self._candidates) - 1
        for index, candidate in enumerate(self._candidates):
            started = self._clock()
            try:
                output = candidate.generator.invoke(messages)
                reply, mood, structured = parse_generation(output)
                if not reply:
                    raise EmptyModelResponse("El modelo devolvió una respuesta vacía.")
            except Exception as exc:  # se clasifica y se traduce a error de dominio
                kind = classify_error(exc)
                attempts.append(
                    Attempt(candidate.model, kind.value, self._elapsed(started))
                )
                if index < last and kind in FALLBACK_KINDS:
                    logger.warning(
                        "Fallo de %s (%s, %.1f s): %s. Se prueba %s.",
                        candidate.model,
                        kind.value,
                        attempts[-1].seconds,
                        describe_error(exc),
                        self._candidates[index + 1].model,
                    )
                    continue
                if index < last:
                    logger.error(
                        "Fallo de %s (%s) que no activa el respaldo: %s",
                        candidate.model,
                        kind.value,
                        describe_error(exc),
                    )
                raise LLMUnavailable(
                    "El modelo de chat no respondió.", attempts=attempts
                ) from exc

            attempts.append(
                Attempt(candidate.model, OUTCOME_OK, self._elapsed(started))
            )
            if index > 0:
                logger.info(
                    "Respondió el respaldo %s tras %d intento(s) fallido(s).",
                    candidate.model,
                    index,
                )
            return Generation(
                reply=reply,
                mood=mood,
                structured=structured,
                model=candidate.model,
                structured_mode=candidate.structured_mode,
                attempts=tuple(attempts),
            )
        raise AssertionError("inalcanzable")  # pragma: no cover

    def _elapsed(self, started: float) -> float:
        return max(0.0, self._clock() - started)


def generator_for(
    llm: BaseChatModel | Runnable[Any, Any], structured_mode: str
) -> Runnable[list[BaseMessage], Any]:
    """Generador `{reply, mood}` del modo indicado (`native` = Gemini)."""
    if structured_mode == "native":
        return build_native_generator(llm)
    return build_structured_generator(llm, structured_mode)


def build_fallback_generator(
    settings: Settings | None = None,
    *,
    model_factory: Callable[[Settings, str], Any] = build_chat_model,
) -> FallbackGenerator:
    """Cadena principal + respaldos según la configuración.

    Construye todos los modelos al arrancar: un respaldo mal configurado (p. ej.
    sin clave) es `LLMConfigurationError` en el arranque, no en mitad de un chat.
    """
    settings = settings or get_settings()
    candidates = []
    for provider in [settings.llm_provider, *settings.llm_fallback_provider_list]:
        mode = str(generation_settings(settings, provider)["structured_mode"])
        candidates.append(
            Candidate(
                model=chat_model_name(settings, provider),
                generator=generator_for(model_factory(settings, provider), mode),
                structured_mode=mode,
            )
        )
    generator = FallbackGenerator(candidates)
    if len(candidates) > 1:
        logger.info("Cadena de chat: %s", " -> ".join(generator.models))
    return generator
