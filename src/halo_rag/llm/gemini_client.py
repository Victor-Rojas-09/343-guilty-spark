"""Cliente base del SDK oficial de Google Gemini (`google-genai`).

Responsabilidad:
    Crear y reutilizar el cliente de Gemini a partir de la configuración.
    Los módulos `embeddings.py` y `generation.py` construyen sobre él.
"""

from google import genai

from halo_rag.config import get_settings


def build_client() -> genai.Client:
    """Construye el cliente de Gemini con la API key configurada.

    Raises:
        RuntimeError: Si `GEMINI_API_KEY` no está definida.
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY no está configurada en el entorno.")
    return genai.Client(api_key=settings.gemini_api_key)
