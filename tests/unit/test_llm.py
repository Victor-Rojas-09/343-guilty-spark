import logging
from types import SimpleNamespace

import pytest

from spark_rag.config import Settings
from spark_rag.rag.errors import LLMConfigurationError
from spark_rag.rag.llm import (
    build_chat_model,
    chat_model_name,
    disable_internal_retry,
    supported_providers,
)


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
            gemini_chat_model="gemini-3.8-flash",
            llm_temperature=0.2,
            llm_max_output_tokens=512,
            llm_timeout_seconds=12,
            llm_max_retries=1,
        )
    )

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.endswith("gemini-3.8-flash")
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


def _generate_content_method(model):
    transport = model.client._transport
    return transport._wrapped_methods[transport.generate_content]


def test_gemini_sin_reintento_interno_del_cliente() -> None:
    model = build_chat_model(_settings(gemini_api_key="k", llm_max_retries=0))

    # Solo queda la política de LangChain (LLM_MAX_RETRIES).
    assert _generate_content_method(model)._retry is None
    assert model.max_retries == 0


@pytest.mark.parametrize(
    "fake_model",
    [
        object(),
        SimpleNamespace(client=SimpleNamespace()),
        SimpleNamespace(
            client=SimpleNamespace(
                _transport=SimpleNamespace(generate_content="gc", _wrapped_methods={})
            )
        ),
        SimpleNamespace(
            client=SimpleNamespace(
                _transport=SimpleNamespace(
                    generate_content="gc", _wrapped_methods={"gc": object()}
                )
            )
        ),
    ],
)
def test_desactivar_reintento_tolera_cambios_de_la_libreria(
    fake_model: object, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="spark_rag.rag.llm"):
        assert disable_internal_retry(fake_model) is False

    assert "No se pudo desactivar el reintento interno de Gemini" in caplog.text


def test_gemini_se_construye_aunque_no_se_pueda_desactivar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("spark_rag.rag.llm.disable_internal_retry", lambda model: False)
    model = build_chat_model(_settings(gemini_api_key="k"))
    assert model.model.endswith("gemini-3.8-flash")
