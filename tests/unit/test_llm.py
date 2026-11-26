import pytest

from spark_rag.config import Settings
from spark_rag.rag.errors import LLMConfigurationError
from spark_rag.rag.llm import build_chat_model, chat_model_name, supported_providers


def _settings(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_sin_api_key_falla_con_mensaje_claro() -> None:
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        build_chat_model(_settings(gemini_api_key=""))


def test_proveedor_desconocido() -> None:
    with pytest.raises(LLMConfigurationError, match="no está soportado"):
        build_chat_model(_settings(llm_provider="otro", gemini_api_key="k"))


def test_modelo_vacio() -> None:
    with pytest.raises(LLMConfigurationError, match="GEMINI_CHAT_MODEL"):
        build_chat_model(_settings(gemini_api_key="k", gemini_chat_model="  "))


def test_gemini_recibe_los_parametros_configurados() -> None:
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = build_chat_model(
        _settings(
            gemini_api_key="clave-de-prueba",
            gemini_chat_model="gemini-2.5-flash",
            llm_temperature=0.2,
            llm_max_output_tokens=512,
            llm_timeout_seconds=12,
            llm_max_retries=1,
        )
    )

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.endswith("gemini-2.5-flash")
    assert model.temperature == 0.2
    assert model.max_output_tokens == 512
    assert model.timeout == 12
    assert model.max_retries == 1


def test_la_clave_no_aparece_en_la_representacion() -> None:
    model = build_chat_model(_settings(gemini_api_key="clave-super-secreta"))
    assert "clave-super-secreta" not in repr(model)


def test_utilidades() -> None:
    assert supported_providers() == ("gemini",)
    assert chat_model_name(_settings(gemini_chat_model="m-1")) == "m-1"
