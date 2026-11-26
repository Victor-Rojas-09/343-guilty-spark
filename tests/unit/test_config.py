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
    assert settings.gemini_chat_model == "gemini-3.8-flash"
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


def test_env_file_sobrescribe_el_modelo_por_defecto(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_CHAT_MODEL=gemini-flash-latest\nLLM_TIMEOUT_SECONDS=45\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)

    assert Settings(_env_file=None).gemini_chat_model == "gemini-3.8-flash"
    assert settings.gemini_chat_model == "gemini-flash-latest"
    assert settings.llm_timeout_seconds == 45


def test_respaldo_desactivado_por_defecto() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_fallback_provider_list == []
    assert settings.llm_fallback_timeout_seconds == 30
    assert settings.llm_primary_cooldown_seconds == 60
    assert settings.openai_compat_base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.openai_compat_api_key.get_secret_value() == ""
    assert settings.openai_compat_chat_model == ""
    assert settings.openai_compat_label == "nvidia"
    assert settings.openai_compat_structured_mode == "auto"
    assert settings.openai_compat_extra_body_dict == {}


def test_lista_de_respaldo_sin_duplicados_ni_principal() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        llm_fallback_providers=" OpenAI_Compat, gemini,,openai_compat ",
    )
    assert settings.llm_fallback_provider_list == ["openai_compat"]


def test_timeout_corto_solo_con_respaldo() -> None:
    sin_respaldo = Settings(_env_file=None, llm_timeout_seconds=90)
    assert sin_respaldo.chat_timeout_seconds("gemini") == 90

    con_respaldo = Settings(
        _env_file=None,
        llm_timeout_seconds=90,
        llm_fallback_providers="openai_compat",
        llm_fallback_timeout_seconds=30,
        openai_compat_timeout_seconds=60,
    )
    assert con_respaldo.chat_timeout_seconds("gemini") == 30
    assert con_respaldo.chat_timeout_seconds("openai_compat") == 60
    # Peor caso acotado: principal corto + respaldo.
    assert (
        con_respaldo.chat_timeout_seconds("gemini")
        + con_respaldo.chat_timeout_seconds("openai_compat")
        <= 90
    )


def test_openai_compat_como_principal_usa_su_timeout() -> None:
    settings = Settings(
        _env_file=None, llm_provider="openai_compat", openai_compat_timeout_seconds=45
    )
    assert settings.llm_fallback_provider_list == []
    assert settings.chat_timeout_seconds("openai_compat") == 45


def test_extra_body_json_valido() -> None:
    settings = Settings(
        _env_file=None,
        openai_compat_extra_body='{"chat_template_kwargs": {"thinking": false}}',
    )
    assert settings.openai_compat_extra_body_dict == {
        "chat_template_kwargs": {"thinking": False}
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"openai_compat_extra_body": "{no es json"},
        {"openai_compat_extra_body": "[1, 2]"},
        {"openai_compat_base_url": "integrate.api.nvidia.com/v1"},
        {"openai_compat_structured_mode": "xml"},
        {"llm_fallback_timeout_seconds": 0},
        {"llm_primary_cooldown_seconds": -1},
        {"openai_compat_timeout_seconds": 0},
    ],
)
def test_rechaza_configuracion_de_respaldo_invalida(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


def test_base_url_y_etiqueta_se_normalizan() -> None:
    settings = Settings(
        _env_file=None,
        openai_compat_base_url=" https://example.test/v1/ ",
        openai_compat_label="  ",
        openai_compat_structured_mode=" PROMPT ",
    )
    assert settings.openai_compat_base_url == "https://example.test/v1"
    assert settings.openai_compat_label == "openai_compat"
    assert settings.openai_compat_structured_mode == "prompt"


def test_clave_openai_compat_no_aparece_en_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "nvapi-FAKEFAKEFAKEFAKE")
    settings = Settings(_env_file=None)

    assert settings.openai_compat_api_key.get_secret_value() == "nvapi-FAKEFAKEFAKEFAKE"
    assert "nvapi-FAKE" not in repr(settings)
    assert "nvapi-FAKE" not in str(settings)
