"""Control de límites de uso de la API de Gemini.

Responsabilidad:
    Centralizar lotes (batching), reintentos con espera exponencial y, si hace
    falta, limitación de velocidad (rate limiting) para la capa gratuita.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitPolicy:
    """Parámetros de control de la API (valores iniciales a validar)."""

    batch_size: int = 16
    max_retries: int = 5
    initial_wait_seconds: float = 1.0
    max_wait_seconds: float = 60.0
