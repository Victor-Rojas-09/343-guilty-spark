"""Generación de respuestas con el modelo de Gemini.

Responsabilidad:
    Enviar el prompt (contexto + pregunta) al modelo generativo y devolver la
    respuesta con sus citas. Debe rechazar responder si el contexto no alcanza.
"""


def generate_answer(question: str, context: str) -> str:
    """Genera una respuesta fundamentada en el contexto recuperado.

    Args:
        question: Pregunta del usuario.
        context: Fragmentos recuperados, incluyendo documento y página.

    Returns:
        Respuesta del modelo con citas o una negativa explícita.

    Raises:
        NotImplementedError: Pendiente de implementación en la Fase 2.
    """
    raise NotImplementedError("Generación pendiente de implementación (Fase 2).")
