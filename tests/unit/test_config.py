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


def test_valores_por_defecto_de_etapa_2() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "gemini"
    assert settings.gemini_chat_model == "gemini-2.5-flash"
    assert settings.retrieval_top_k == 5
    assert 0.0 <= settings.retrieval_min_similarity <= 1.0
    assert settings.retrieval_category_list == []
    assert settings.chat_history_max_messages == 10
    assert settings.chat_message_max_chars == 2000


def test_proveedor_se_normaliza() -> None:
    settings = Settings(_env_file=None, llm_provider="  Gemini ")
    assert settings.llm_provider == "gemini"


def test_categorias_desde_csv() -> None:
    settings = Settings(
        _env_file=None, retrieval_categories=" History, dialogs,,history "
    )
    assert settings.retrieval_category_list == ["history", "dialogs"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retrieval_min_similarity": 1.5},
        {"retrieval_min_similarity": -0.1},
        {"retrieval_top_k": 0},
        {"chat_history_max_messages": -1},
        {"chat_message_max_chars": 0},
        {"llm_temperature": 3},
    ],
)
def test_rechaza_parametros_fuera_de_rango(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


def test_lee_variables_de_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "otro-modelo")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "8")
    monkeypatch.setenv("RETRIEVAL_CATEGORIES", "community")
    settings = Settings(_env_file=None)

    assert settings.gemini_chat_model == "otro-modelo"
    assert settings.retrieval_top_k == 8
    assert settings.retrieval_category_list == ["community"]
