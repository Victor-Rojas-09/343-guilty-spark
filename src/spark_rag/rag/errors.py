"""Excepciones de dominio del motor RAG.

La capa HTTP las traduce a códigos y mensajes controlados; sus textos no deben
incluir credenciales, cadenas de conexión ni trazas.
"""

from __future__ import annotations


class SparkError(Exception):
    """Base de los errores del motor conversacional."""


class LLMConfigurationError(SparkError):
    """El proveedor o modelo de chat no está configurado correctamente."""


class LLMUnavailable(SparkError):
    """El modelo de chat falló, agotó la cuota o no respondió a tiempo."""


class RetrievalUnavailable(SparkError):
    """No se pudo consultar la memoria documental (pgvector o embeddings)."""


class SessionNotFound(SparkError):
    """El `session_id` recibido no corresponde a ninguna sesión."""


class StorageUnavailable(SparkError):
    """No se pudo leer o guardar el historial de la sesión."""
