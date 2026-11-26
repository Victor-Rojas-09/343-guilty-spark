import argparse
import json
from pathlib import Path

import pytest

from spark_rag.rag.chain import RagResult
from spark_rag.rag.evaluation import (
    MIN_PAUSE_SECONDS,
    EvalQuestion,
    EvalSet,
    evaluate,
    load_eval_set,
    pause_seconds_arg,
    run_checks,
    summarize,
)
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
