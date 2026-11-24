"""Pruebas de humo del scaffolding (no requieren dependencias externas)."""

from halo_rag import __version__
from halo_rag.config import Settings


def test_version_defined() -> None:
    assert __version__


def test_default_database_url() -> None:
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("postgresql://")
    assert settings.postgres_db == "halo_rag"
