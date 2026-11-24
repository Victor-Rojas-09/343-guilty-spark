"""Pipeline RAG completo: pregunta -> recuperación -> generación con citas.

Responsabilidad:
    Coordinar `retrieval`, `llm.embeddings` y `llm.generation`, aplicando el
    prompt del sistema y gestionando el caso de contexto insuficiente.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGAnswer:
    """Resultado del pipeline, incluyendo las fuentes utilizadas."""

    answer: str
    sources: list[str]


class RAGPipeline:
    """Orquesta las etapas de recuperación y generación."""

    def answer(self, question: str) -> RAGAnswer:
        """Responde una pregunta con citas del manual.

        Raises:
            NotImplementedError: Pendiente de implementación en la Fase 2.
        """
        raise NotImplementedError("Pipeline RAG pendiente de implementación (Fase 2).")
