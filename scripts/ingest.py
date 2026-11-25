#!/usr/bin/env python3
"""Punto de entrada CLI de la ingesta documental offline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spark_rag.config import get_settings
from spark_rag.ingestion.pipeline import IngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingiere PDF, Markdown y TXT del corpus en PostgreSQL/pgvector."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directorio del corpus (por defecto DATA_DIR, p. ej. data/).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Nombre de la colección vectorial (por defecto COLLECTION_NAME).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Nivel de logging: DEBUG, INFO, WARNING, ERROR.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    updates: dict[str, object] = {}
    if args.collection:
        updates["collection_name"] = args.collection
    if args.log_level:
        updates["log_level"] = args.log_level.upper()
    if updates:
        settings = settings.model_copy(update=updates)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        report = IngestionPipeline(settings).run(args.data_dir)
    except Exception as exc:  # noqa: BLE001 - frontera CLI
        logging.getLogger(__name__).error("La ingesta no pudo ejecutarse: %s", exc)
        return 2

    print(report.summary())
    for result in report.results:
        detail = f" ({result.chunk_count} chunks)" if result.chunk_count else ""
        error = f" -> {result.error}" if result.error else ""
        print(
            f"[{result.status:7}] {result.source}{detail} "
            f"{result.elapsed_seconds:.2f}s{error}"
        )

    return 1 if report.files_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
