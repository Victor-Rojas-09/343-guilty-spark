"""Configuración central del proyecto.

Responsabilidad:
    Leer y validar las variables de entorno definidas en `.env` (en desarrollo)
    o en el entorno del proceso (CI/producción). Ningún otro módulo debe leer
    `os.environ` directamente.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración tipada del sistema RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL
    postgres_user: str = "halo"
    postgres_password: str = "halo_dev_password"
    postgres_db: str = "halo_rag"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Google Gemini
    gemini_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_generation_model: str = "gemini-2.5-flash"

    # Parámetros RAG (valores iniciales; ver README)
    embedding_dim: int = 768
    chunk_size: int = 800
    chunk_overlap: int = 150
    pdf_path: str = "data/halo_combat_evolved_manual.pdf"

    # Logging
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        """URL de conexión compatible con psycopg/SQLAlchemy."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia única de la configuración."""
    return Settings()
