"""Configuración central del proyecto.

Toda la lectura de variables de entorno pasa por este módulo. El resto de la
aplicación debe consumir `get_settings()` y nunca leer `os.environ` de forma
directa.

Las variables se cargan desde el `.env` de la raíz del repositorio cuando se
ejecuta en local. En Docker/CI se inyectan como variables de entorno del
proceso.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuración tipada y validada del sistema."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL
    postgres_user: str = "spark"
    postgres_password: str = "spark_dev_password"
    postgres_db: str = "spark_rag"
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)

    # API
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)

    # Google Gemini
    gemini_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = Field(default=768, ge=128, le=3072)

    # Ingesta
    data_dir: Path = Path("data")
    collection_name: str = "spark_documents"
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)

    # Logging
    log_level: str = "INFO"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_menor_que_chunk(cls, value: int, info: object) -> int:
        chunk_size = getattr(info, "data", {}).get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE.")
        return value

    @property
    def database_url(self) -> str:
        """URL de conexión para psycopg (driver síncrono)."""
        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password, safe="")
        return (
            f"postgresql://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sqlalchemy_database_url(self) -> str:
        """URL de conexión que usa SQLAlchemy/psycopg3 para LangChain."""
        return self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    def resolve_data_dir(self) -> Path:
        """Resuelve `DATA_DIR` contra la raíz del proyecto si es relativa."""
        if self.data_dir.is_absolute():
            return self.data_dir
        return (PROJECT_ROOT / self.data_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia única de la configuración."""
    return Settings()
