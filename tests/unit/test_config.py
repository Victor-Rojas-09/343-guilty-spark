from pathlib import Path

import pytest
from pydantic import ValidationError

from spark_rag.config import Settings


def test_urls_de_conexion() -> None:
    settings = Settings(
        _env_file=None,
        postgres_user="user",
        postgres_password="p@ss",
        postgres_db="spark",
        postgres_host="db",
        postgres_port=6543,
    )

    assert settings.database_url == "postgresql://user:p%40ss@db:6543/spark"
    assert settings.sqlalchemy_database_url == (
        "postgresql+psycopg://user:p%40ss@db:6543/spark"
    )


def test_data_dir_se_resuelve_contra_la_raiz() -> None:
    settings = Settings(_env_file=None, data_dir=Path("data"))
    assert settings.resolve_data_dir().is_absolute()
    assert settings.resolve_data_dir().name == "data"


def test_overlap_debe_ser_menor_que_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)
