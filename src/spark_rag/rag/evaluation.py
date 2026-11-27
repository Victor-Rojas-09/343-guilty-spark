"""Evaluación del personaje y registro de resultados (manual §4.2-§4.4).

Las comprobaciones automáticas son heurísticas baratas (categorías
recuperadas, marcadores de laguna, menciones esperadas, longitud máxima). No
sustituyen la revisión humana del comportamiento esperado: cada resultado
lleva `manual_review = "pendiente"` para completarlo a mano.

Cada resultado indica qué modelo respondió (principal o respaldo), los intentos
de la cadena de respaldo y si la salida fue estructurada. El criterio de cierre
«Gemini genera respuestas con la personalidad acordada» solo puede avanzar con
respuestas de Gemini: las de otro proveedor validan el pipeline, no ese
criterio (ver `compliance`).
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spark_rag.rag.chain import RagResult
from spark_rag.rag.history import ChatTurn

MIN_PAUSE_SECONDS = 15.0
"""Espera mínima entre llamadas reales a Gemini.

El plan gratuito de `gemini-3.8-flash` admite 5 peticiones por minuto; 15 s de
separación deja margen para un reintento interno del cliente.
"""


def pause_seconds_arg(value: str) -> float:
    """Tipo de argparse para `--pause`: exige al menos `MIN_PAUSE_SECONDS`."""
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{value}' no es un número") from exc
    if seconds < MIN_PAUSE_SECONDS:
        raise argparse.ArgumentTypeError(
            f"la pausa mínima entre llamadas es {MIN_PAUSE_SECONDS:g} s"
        )
    return seconds


@dataclass(frozen=True)
class EvalQuestion:
    """Pregunta del set de evaluación."""

    id: str
    coverage: str
    question: str
    expected_behavior: str
    expected_categories: list[str] = field(default_factory=list)
    expect_evidence: bool = False
    expect_gap: bool = False
    mentions_any: list[str] = field(default_factory=list)
    forbid_any: list[str] = field(default_factory=list)
    max_reply_chars: int | None = None
    history_from: str | None = None


@dataclass(frozen=True)
class EvalSet:
    version: str
    questions: list[EvalQuestion]
    gap_markers: list[str]


def load_eval_set(path: Path) -> EvalSet:
    """Carga y valida el archivo JSON de preguntas."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = [EvalQuestion(**item) for item in raw["questions"]]
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(f"Id de pregunta duplicado: {question.id}")
        if question.history_from and question.history_from not in seen:
            raise ValueError(
                f"{question.id}: history_from debe apuntar a una pregunta anterior."
            )
        seen.add(question.id)
    return EvalSet(
        version=str(raw.get("version", "sin_version")),
        questions=questions,
        gap_markers=list(raw.get("gap_markers", [])),
    )


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _contains_any(text: str, needles: list[str]) -> bool:
    haystack = _normalize(text)
    return any(_normalize(needle) in haystack for needle in needles)


def run_checks(
    question: EvalQuestion, result: RagResult, gap_markers: list[str]
) -> dict[str, bool]:
    """Evalúa las comprobaciones automáticas aplicables a una respuesta."""
    checks: dict[str, bool] = {"reply_not_empty": bool(result.reply.strip())}
    categories = {detail["category"] for detail in result.source_details}

    if question.expect_evidence:
        checks["has_sources"] = bool(result.sources)
    if question.expected_categories:
        checks["expected_category_retrieved"] = bool(
            categories & set(question.expected_categories)
        )
    if question.expect_gap:
        checks["declares_gap"] = _contains_any(result.reply, gap_markers)
    if question.mentions_any:
        checks["mentions_expected"] = _contains_any(result.reply, question.mentions_any)
    if question.forbid_any:
        checks["no_forbidden_claims"] = not _contains_any(
            result.reply, question.forbid_any
        )
    if question.max_reply_chars is not None:
        checks["reply_length_ok"] = len(result.reply) <= question.max_reply_chars
    return checks


NO_MODEL = "sin_modelo"
"""Clave del resumen por modelo para preguntas sin respuesta (ERROR)."""

GEMINI_PERSONALITY_CRITERION = "Gemini genera respuestas con la personalidad acordada"


def _attempt_dicts(attempts: Iterable[object]) -> list[dict[str, Any]]:
    """Intentos serializables (`Attempt.as_dict`), sin mensajes de error."""
    items: list[dict[str, Any]] = []
    for attempt in attempts:
        as_dict = getattr(attempt, "as_dict", None)
        if callable(as_dict):
            items.append(as_dict())
    return items


def evaluate(
    eval_set: EvalSet,
    answer: Callable[[str, list[ChatTurn]], RagResult],
    *,
    pause_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Ejecuta cada pregunta y devuelve los resultados serializables.

    `answer(question, history)` debe ejecutar la cadena RAG. Un error en una
    pregunta se registra como `ERROR` y no detiene el resto.
    """
    results: list[dict[str, Any]] = []
    transcripts: dict[str, list[ChatTurn]] = {}

    for index, question in enumerate(eval_set.questions):
        if index and pause_seconds > 0:
            time.sleep(pause_seconds)
        history = list(transcripts.get(question.history_from or "", []))
        started = time.monotonic()
        record: dict[str, Any] = {
            "id": question.id,
            "coverage": question.coverage,
            "question": question.question,
            "expected_behavior": question.expected_behavior,
            "history_from": question.history_from,
            "manual_review": "pendiente",
        }
        try:
            result = answer(question.question, history)
        except Exception as exc:  # noqa: BLE001 - se registra y se continúa
            # `LLMUnavailable` trae los intentos de la cadena (modelo y tipo de fallo).
            attempts = _attempt_dicts(getattr(exc, "attempts", ()))
            record.update(
                verdict="ERROR",
                error=type(exc).__name__,
                model=None,
                fallback_used=len(attempts) > 1,
                attempts=attempts,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
            results.append(record)
            continue

        checks = run_checks(question, result, eval_set.gap_markers)
        transcripts[question.id] = [
            *history,
            ChatTurn(role="user", content=question.question),
            ChatTurn(role="assistant", content=result.reply),
        ]
        record.update(
            model=result.model or None,
            fallback_used=result.fallback_used,
            structured_mode=result.structured_mode or None,
            attempts=_attempt_dicts(result.attempts),
            reply=result.reply,
            mood=result.mood.value,
            structured_output=result.structured,
            sources=result.sources,
            source_details=result.source_details,
            checks=checks,
            verdict="PASS" if all(checks.values()) else "FAIL",
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
        results.append(record)
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    """Cuenta veredictos."""
    summary = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    for record in results:
        summary[record["verdict"]] = summary.get(record["verdict"], 0) + 1
    return summary


def summarize_by_model(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """PASS/FAIL/ERROR por modelo que respondió (`sin_modelo` para los ERROR)."""
    by_model: dict[str, dict[str, int]] = {}
    for record in results:
        model = record.get("model") or NO_MODEL
        counts = by_model.setdefault(model, {"PASS": 0, "FAIL": 0, "ERROR": 0})
        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
    return by_model


def compliance(
    results: list[dict[str, Any]], *, gemini_model: str
) -> list[dict[str, Any]]:
    """Estado de los criterios de cierre que dependen del modelo que respondió.

    «Gemini genera respuestas con la personalidad acordada»:
    - `PENDIENTE` si ninguna respuesta de esta pasada es de Gemini (p. ej. todas
      de NVIDIA): esas respuestas validan el pipeline, no este criterio.
    - `REVISION_MANUAL` si hay respuestas de Gemini: la personalidad se confirma
      a mano (`manual_review`); el informe nunca lo da por cumplido solo.
    """
    answered = [r for r in results if r.get("model")]
    gemini = [r for r in answered if r["model"] == gemini_model]
    others = len(answered) - len(gemini)
    if gemini:
        passed = sum(1 for r in gemini if r["verdict"] == "PASS")
        status = "REVISION_MANUAL"
        detail = (
            f"{len(gemini)} respuesta(s) de {gemini_model} ({passed} PASS "
            "automático); falta confirmar la personalidad en manual_review."
        )
    else:
        status = "PENDIENTE"
        detail = (
            f"Ninguna respuesta de {gemini_model} en esta pasada; {others} "
            "respuesta(s) de otros modelos validan el pipeline, no este criterio."
        )
    return [
        {
            "criterion": GEMINI_PERSONALITY_CRITERION,
            "status": status,
            "gemini_answers": len(gemini),
            "other_model_answers": others,
            "detail": detail,
        }
    ]
