"""Punto de entrada CLI para preguntar al sistema RAG."""

import argparse


def main() -> None:
    """Responde una pregunta en lenguaje natural (Fase 2)."""
    parser = argparse.ArgumentParser(description="Pregunta sobre el manual de Halo: Combat Evolved.")
    parser.add_argument("question", help="Pregunta en lenguaje natural.")
    parser.parse_args()
    raise SystemExit("Consulta pendiente de implementación (Fase 2).")


if __name__ == "__main__":
    main()
