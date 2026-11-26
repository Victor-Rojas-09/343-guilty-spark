#!/usr/bin/env python3
"""Evalúa al personaje con Gemini real y registra los resultados (§4.3-§4.4).

Requiere el corpus ya ingerido y `GEMINI_API_KEY`. No usa sesiones de la base
de datos: cada pregunta se responde con la cadena RAG y las preguntas con
`history_from` reciben el historial del turno indicado.

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --questions tests/eval/spark_questions.json \
        --output-dir reports/eval --pause 20
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spark_rag import __version__
from spark_rag.config import get_settings
from spark_rag.db.connection import check_database, get_connection
from spark_rag.rag.chain import build_chain
from spark_rag.rag.evaluation import (
    MIN_PAUSE_SECONDS,
    evaluate,
    load_eval_set,
    pause_seconds_arg,
    summarize,
)
from spark_rag.rag.llm import chat_model_name, generation_settings, get_chat_model
from spark_rag.rag.prompts import PROMPT_VERSION
from spark_rag.rag.retriever import build_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación del personaje Spark.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "tests" / "eval" / "spark_questions.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "eval"
    )
    parser.add_argument(
        "--pause",
        type=pause_seconds_arg,
        default=MIN_PAUSE_SECONDS,
        help=(
            "Segundos de espera entre preguntas para respetar el límite por "
            f"minuto de Gemini (mínimo y valor por defecto: {MIN_PAUSE_SECONDS:g})."
        ),
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def git_commit() -> str:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        suffix = "+cambios" if dirty.stdout.strip() else ""
        return output.stdout.strip() + suffix
    except Exception:  # noqa: BLE001 - p. ej. dentro de Docker sin .git
        return "desconocido"


def indexed_documents() -> list[dict[str, object]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT source, sha256, chunk_count
            FROM ingested_files
            WHERE status = 'indexed'
            ORDER BY source
            """
        ).fetchall()
    return [
        {
            "source": row["source"],
            "sha256": (row["sha256"] or "")[:12],
            "chunks": row["chunk_count"],
        }
        for row in rows
    ]


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper())
    settings = get_settings()

    status = check_database()
    if not status.ok:
        print(f"PostgreSQL/pgvector no disponible: {status.error}", file=sys.stderr)
        return 2

    eval_set = load_eval_set(args.questions)
    retriever = build_retriever(settings)
    chain = build_chain(retrieve=retriever.retrieve, llm=get_chat_model())

    started_at = datetime.now(UTC)
    results = evaluate(
        eval_set,
        lambda question, history: chain.invoke(
            question,
            history=history,
            retrieval_query=(
                f"{history[-2].content}\n{question}" if len(history) >= 2 else question
            ),
        ),
        pause_seconds=args.pause,
    )
    summary = summarize(results)

    report = {
        "run": {
            "date": started_at.isoformat(timespec="seconds"),
            "code_version": __version__,
            "git_commit": git_commit(),
            "prompt_version": PROMPT_VERSION,
            "eval_set_version": eval_set.version,
            "chat_provider": settings.llm_provider,
            "chat_model": chat_model_name(settings),
            "temperature": generation_settings(settings)["temperature"],
            # Modelo, temperature, top_p y structured_mode de cada proveedor de la
            # cadena (principal y respaldos), sin credenciales.
            "chat_generation": [
                generation_settings(settings, provider)
                for provider in [
                    settings.llm_provider,
                    *settings.llm_fallback_provider_list,
                ]
            ],
            "pause_seconds": args.pause,
            "embedding_model": settings.gemini_embedding_model,
            "embedding_dim": settings.embedding_dim,
            "collection": settings.collection_name,
            "retrieval": {
                "top_k": settings.retrieval_top_k,
                "min_similarity": settings.retrieval_min_similarity,
                "categories": settings.retrieval_category_list,
            },
            "indexed_documents": indexed_documents(),
            "summary": summary,
        },
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"eval_{stamp}_{PROMPT_VERSION}.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    for record in results:
        failed = [name for name, ok in record.get("checks", {}).items() if not ok]
        extra = f" fallos={failed}" if failed else ""
        error = f" error={record['error']}" if "error" in record else ""
        print(f"[{record['verdict']}] {record['id']}{extra}{error}")
    print(
        f"PASS={summary['PASS']} FAIL={summary['FAIL']} ERROR={summary['ERROR']} "
        f"prompt={PROMPT_VERSION} modelo={chat_model_name(settings)}"
    )
    print(f"Informe: {output}")
    return 0 if summary["ERROR"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
