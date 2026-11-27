import argparse
import json
from pathlib import Path

import pytest

from spark_rag.rag.chain import RagResult
from spark_rag.rag.errors import LLMUnavailable
from spark_rag.rag.evaluation import (
    MIN_PAUSE_SECONDS,
    EvalQuestion,
    EvalSet,
    compliance,
    evaluate,
    load_eval_set,
    pause_seconds_arg,
    run_checks,
    summarize,
    summarize_by_model,
)
from spark_rag.rag.fallback import Attempt
from spark_rag.rag.mood import Mood
from spark_rag.rag.sources import RetrievedChunk

EVAL_FILE = Path(__file__).resolve().parents[1] / "eval" / "spark_questions.json"

COMMUNITY = RetrievedChunk(
    content="Debate",
    similarity=0.75,
    metadata={"source": "community/07.pdf", "category": "community", "page": 1},
)


def _result(reply: str, chunks: list[RetrievedChunk] | None = None) -> RagResult:
    return RagResult(reply=reply, mood=Mood.THINKING, chunks=chunks or [])


def test_set_de_evaluacion_valido_y_completo() -> None:
    eval_set = load_eval_set(EVAL_FILE)
    coverages = " ".join(q.coverage for q in eval_set.questions)

    for required in (
        "Identidad",
        "Protocolo",
        "Aislamiento",
        "Halo 3",
        "Evolución",
        "Fuentes",
        "Laguna",
        "Diálogo",
    ):
        assert required in coverages
    assert any(q.expect_gap for q in eval_set.questions)
    assert eval_set.gap_markers


def test_rechaza_ids_duplicados_e_historial_invalido(tmp_path: Path) -> None:
    base = {"coverage": "c", "question": "q", "expected_behavior": "e"}
    path = tmp_path / "eval.json"

    path.write_text(json.dumps({"questions": [{"id": "a", **base}] * 2}))
    with pytest.raises(ValueError, match="duplicado"):
        load_eval_set(path)

    path.write_text(
        json.dumps({"questions": [{"id": "a", "history_from": "b", **base}]})
    )
    with pytest.raises(ValueError, match="history_from"):
        load_eval_set(path)


def test_comprobaciones_de_fuentes_comunitarias() -> None:
    question = EvalQuestion(
        id="r",
        coverage="Fuentes",
        question="¿Reddit es canon?",
        expected_behavior="",
        expected_categories=["community"],
        expect_evidence=True,
        mentions_any=["comunidad"],
    )
    ok = run_checks(question, _result("Es un debate de la COMUNIDAD.", [COMMUNITY]), [])
    assert all(ok.values())

    bad = run_checks(question, _result("Es canon.", []), [])
    assert bad["has_sources"] is False
    assert bad["expected_category_retrieved"] is False
    assert bad["mentions_expected"] is False


def test_comprobacion_de_laguna_ignora_acentos() -> None:
    question = EvalQuestion(
        id="g",
        coverage="Laguna",
        question="?",
        expected_behavior="",
        expect_gap=True,
        forbid_any=["Voss fue"],
        max_reply_chars=100,
    )
    checks = run_checks(
        question,
        _result("Mis protocolos no registran datos concluyentes, Reclamador."),
        ["NO REGISTRAN"],
    )
    assert checks == {
        "reply_not_empty": True,
        "declares_gap": True,
        "no_forbidden_claims": True,
        "reply_length_ok": True,
    }


def test_evaluate_encadena_historial_y_registra_errores() -> None:
    eval_set = EvalSet(
        version="t",
        gap_markers=[],
        questions=[
            EvalQuestion(id="a", coverage="c", question="uno", expected_behavior=""),
            EvalQuestion(
                id="b",
                coverage="c",
                question="dos",
                expected_behavior="",
                history_from="a",
            ),
            EvalQuestion(id="c", coverage="c", question="tres", expected_behavior=""),
        ],
    )
    seen_history: dict[str, list[str]] = {}

    def answer(question, history):
        seen_history[question] = [turn.content for turn in history]
        if question == "tres":
            raise RuntimeError("cuota")
        return _result(f"respuesta {question}")

    results = evaluate(eval_set, answer)

    assert seen_history["dos"] == ["uno", "respuesta uno"]
    assert [r["verdict"] for r in results] == ["PASS", "PASS", "ERROR"]
    assert results[2]["error"] == "RuntimeError"
    assert results[0]["manual_review"] == "pendiente"
    assert summarize(results) == {"PASS": 2, "FAIL": 0, "ERROR": 1}


def test_pausa_minima_entre_llamadas() -> None:
    assert MIN_PAUSE_SECONDS == 15
    assert pause_seconds_arg("15") == 15
    assert pause_seconds_arg("20.5") == 20.5
    for value in ("5", "0", "-1", "abc"):
        with pytest.raises(argparse.ArgumentTypeError):
            pause_seconds_arg(value)


def test_evaluate_espera_entre_preguntas(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("spark_rag.rag.evaluation.time.sleep", sleeps.append)
    eval_set = EvalSet(
        version="t",
        gap_markers=[],
        questions=[
            EvalQuestion(id=q, coverage="c", question=q, expected_behavior="")
            for q in ("a", "b", "c")
        ],
    )

    evaluate(eval_set, lambda q, h: _result(q), pause_seconds=MIN_PAUSE_SECONDS)

    # Sin espera antes de la primera; 15 s antes de cada una de las siguientes.
    assert sleeps == [MIN_PAUSE_SECONDS, MIN_PAUSE_SECONDS]


GEMINI = "gemini-3.8-flash"
NVIDIA = "nvidia:mistralai/mistral-large-4-1t-a52b"


def _three_questions() -> EvalSet:
    return EvalSet(
        version="t",
        gap_markers=[],
        questions=[
            EvalQuestion(id=q, coverage="c", question=q, expected_behavior="")
            for q in ("a", "b", "c")
        ],
    )


def _answer_with_models(question: str, history: list) -> RagResult:
    if question == "a":
        return RagResult(
            reply="Saludos.",
            mood=Mood.IDLE,
            model=GEMINI,
            structured_mode="native",
            attempts=(Attempt(GEMINI, "ok", 2.5),),
        )
    if question == "b":
        return RagResult(
            reply="Respuesta del respaldo.",
            mood=Mood.THINKING,
            structured=False,
            model=NVIDIA,
            structured_mode="function_calling",
            attempts=(Attempt(GEMINI, "unavailable", 30.0), Attempt(NVIDIA, "ok", 4.0)),
        )
    raise LLMUnavailable(
        attempts=(Attempt(GEMINI, "rate_limit", 0.4), Attempt(NVIDIA, "timeout", 60.0))
    )


def test_cada_pregunta_registra_el_modelo_y_los_intentos() -> None:
    a, b, c = evaluate(_three_questions(), _answer_with_models)

    assert (a["model"], a["fallback_used"], a["structured_mode"]) == (
        GEMINI,
        False,
        "native",
    )
    assert a["structured_output"] is True
    assert a["attempts"] == [{"model": GEMINI, "outcome": "ok", "seconds": 2.5}]

    assert (b["model"], b["fallback_used"], b["structured_output"]) == (
        NVIDIA,
        True,
        False,
    )
    assert [x["outcome"] for x in b["attempts"]] == ["unavailable", "ok"]

    assert c["verdict"] == "ERROR"
    assert c["error"] == "LLMUnavailable"
    assert c["model"] is None
    assert c["fallback_used"] is True
    assert c["attempts"] == [
        {"model": GEMINI, "outcome": "rate_limit", "seconds": 0.4},
        {"model": NVIDIA, "outcome": "timeout", "seconds": 60.0},
    ]


def test_error_sin_intentos_no_rompe_el_registro() -> None:
    def answer(question, history):
        raise RuntimeError("fallo de recuperación")

    (record, *_) = evaluate(_three_questions(), answer)
    assert record["attempts"] == []
    assert record["fallback_used"] is False


def test_el_registro_es_serializable_a_json() -> None:
    results = evaluate(_three_questions(), _answer_with_models)
    assert json.loads(json.dumps(results, ensure_ascii=False))


def test_resumen_por_modelo() -> None:
    results = evaluate(_three_questions(), _answer_with_models)

    assert summarize_by_model(results) == {
        GEMINI: {"PASS": 1, "FAIL": 0, "ERROR": 0},
        NVIDIA: {"PASS": 1, "FAIL": 0, "ERROR": 0},
        "sin_modelo": {"PASS": 0, "FAIL": 0, "ERROR": 1},
    }


def test_criterio_gemini_pendiente_si_solo_responde_nvidia() -> None:
    results = [
        {"model": NVIDIA, "verdict": "PASS"},
        {"model": NVIDIA, "verdict": "PASS"},
        {"model": None, "verdict": "ERROR"},
    ]

    (row,) = compliance(results, gemini_model=GEMINI)

    assert row["criterion"] == "Gemini genera respuestas con la personalidad acordada"
    assert row["status"] == "PENDIENTE"
    assert (row["gemini_answers"], row["other_model_answers"]) == (0, 2)
    assert "validan el pipeline" in row["detail"]


def test_criterio_gemini_pendiente_si_todo_es_error() -> None:
    (row,) = compliance([{"model": None, "verdict": "ERROR"}], gemini_model=GEMINI)
    assert row["status"] == "PENDIENTE"


def test_criterio_gemini_nunca_se_cumple_automaticamente() -> None:
    results = [
        {"model": GEMINI, "verdict": "PASS"},
        {"model": GEMINI, "verdict": "FAIL"},
        {"model": NVIDIA, "verdict": "PASS"},
    ]

    (row,) = compliance(results, gemini_model=GEMINI)

    assert row["status"] == "REVISION_MANUAL"
    assert (row["gemini_answers"], row["other_model_answers"]) == (2, 1)
    assert "1 PASS" in row["detail"]
