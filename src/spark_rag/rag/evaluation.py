"""Evaluación del personaje y registro de resultados (manual §4.2-§4.4).

Las comprobaciones automáticas son heurísticas baratas (categorías
recuperadas, marcadores de laguna, menciones esperadas, longitud máxima). No
sustituyen la revisión humana del comportamiento esperado: cada resultado
lleva `manual_review = "pendiente"` para completarlo a mano.
"""

from __future__ import annotations

import json
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spark_rag.rag.chain import RagResult
from spark_rag.rag.history import ChatTurn


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
            record.update(
                verdict="ERROR",
                error=type(exc).__name__,
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
