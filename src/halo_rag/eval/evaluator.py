"""Arnés de evaluación del sistema RAG.

Responsabilidad:
    Cargar el conjunto de preguntas de `eval/questions.example.json`, ejecutar
    el pipeline y calcular métricas de calidad y de abstención correcta.
"""

from pathlib import Path
from typing import Any


def run_evaluation(questions_path: Path) -> dict[str, Any]:
    """Ejecuta el conjunto de evaluación y devuelve las métricas.

    Raises:
        NotImplementedError: Pendiente de implementación en la Fase 3.
    """
    raise NotImplementedError("Evaluación pendiente de implementación (Fase 3).")
