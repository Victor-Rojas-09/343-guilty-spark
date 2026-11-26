import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from spark_rag.config import PROJECT_ROOT, Settings
from spark_rag.rag.errors import LLMConfigurationError
from spark_rag.rag.llm import (
    build_chat_model,
    chat_model_name,
    disable_internal_retry,
    generation_settings,
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
    assert supported_providers() == ("gemini", "openai_compat")
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


FAKE_KEY = "nvapi-FAKEFAKEFAKEFAKE"


def _compat(**kwargs: object) -> Settings:
    values: dict[str, object] = {
        "llm_provider": "openai_compat",
        "openai_compat_api_key": FAKE_KEY,
        "openai_compat_chat_model": "deepseek-ai/modelo-falso",
    }
    values.update(kwargs)
    return _settings(**values)


def test_openai_compat_recibe_los_parametros_configurados() -> None:
    from langchain_core.messages import HumanMessage
    from langchain_openai.chat_models.base import BaseChatOpenAI

    model = build_chat_model(
        _compat(
            openai_compat_base_url="https://example.test/v1",
            llm_temperature=0.3,
            openai_compat_max_output_tokens=777,
            openai_compat_timeout_seconds=42,
            llm_max_retries=0,
            openai_compat_extra_body='{"chat_template_kwargs": {"thinking": false}}',
        )
    )

    assert type(model) is BaseChatOpenAI
    assert model.model_name == "deepseek-ai/modelo-falso"
    assert model.openai_api_base == "https://example.test/v1"
    assert model.temperature == 0.3
    assert model.request_timeout == 42
    assert model.max_retries == 0
    # Se envía `max_tokens` tal cual (no `max_completion_tokens`) y el extra_body.
    payload = model._get_request_payload([HumanMessage(content="hola")])
    assert payload["max_tokens"] == 777
    assert "max_completion_tokens" not in payload
    assert payload["extra_body"] == {"chat_template_kwargs": {"thinking": False}}


def test_openai_compat_como_principal_sin_clave_de_gemini() -> None:
    settings = _compat(gemini_api_key="")
    model = build_chat_model(settings)

    assert model.model_name == "deepseek-ai/modelo-falso"
    assert chat_model_name(settings) == "nvidia:deepseek-ai/modelo-falso"


def test_proveedor_explicito_ignora_llm_provider() -> None:
    settings = _compat(llm_provider="gemini", gemini_api_key="k")

    assert build_chat_model(settings, provider="openai_compat").model_name == (
        "deepseek-ai/modelo-falso"
    )
    assert chat_model_name(settings, provider="openai_compat") == (
        "nvidia:deepseek-ai/modelo-falso"
    )
    assert chat_model_name(settings) == "gemini-3.8-flash"


def test_openai_compat_sin_clave() -> None:
    with pytest.raises(LLMConfigurationError, match="OPENAI_COMPAT_API_KEY"):
        build_chat_model(_compat(openai_compat_api_key=" "))


def test_openai_compat_sin_modelo_no_expone_la_clave() -> None:
    with pytest.raises(LLMConfigurationError, match="OPENAI_COMPAT_CHAT_MODEL") as info:
        build_chat_model(_compat(openai_compat_chat_model=""))
    assert FAKE_KEY not in str(info.value)


def test_la_clave_openai_compat_no_aparece_en_la_representacion() -> None:
    model = build_chat_model(_compat())
    assert FAKE_KEY not in repr(model)
    assert FAKE_KEY not in str(model)


def test_sin_langchain_openai_da_error_claro(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langchain_openai.chat_models.base", None)
    with pytest.raises(LLMConfigurationError, match="langchain-openai"):
        build_chat_model(_compat())


def test_importacion_perezosa_de_langchain_openai() -> None:
    code = (
        "import sys\n"
        "from spark_rag.config import Settings\n"
        "from spark_rag.rag.llm import build_chat_model\n"
        "build_chat_model(Settings(_env_file=None, gemini_api_key='k'))\n"
        "print('langchain_openai' in sys.modules, 'openai' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
        timeout=120,
        check=True,
    )
    assert result.stdout.strip() == "False False"


def test_gemini_usa_el_timeout_corto_solo_con_respaldo() -> None:
    sin_respaldo = build_chat_model(
        _settings(gemini_api_key="k", llm_timeout_seconds=90)
    )
    con_respaldo = build_chat_model(
        _settings(
            gemini_api_key="k",
            llm_timeout_seconds=90,
            llm_fallback_providers="openai_compat",
            llm_fallback_timeout_seconds=30,
        )
    )
    assert sin_respaldo.timeout == 90
    assert con_respaldo.timeout == 30


def _payload(model) -> dict:
    from langchain_core.messages import HumanMessage

    return model._get_request_payload([HumanMessage(content="hola")])


def test_openai_compat_con_muestreo_propio() -> None:
    model = build_chat_model(
        _compat(
            llm_temperature=0.4, openai_compat_temperature=1.0, openai_compat_top_p=0.95
        )
    )
    payload = _payload(model)

    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95


def test_openai_compat_sin_muestreo_propio_usa_llm_temperature() -> None:
    payload = _payload(build_chat_model(_compat(llm_temperature=0.4)))

    assert payload["temperature"] == 0.4
    assert "top_p" not in payload


def test_gemini_no_usa_el_muestreo_de_openai_compat() -> None:
    model = build_chat_model(
        _settings(
            gemini_api_key="k",
            llm_temperature=0.4,
            openai_compat_temperature=1.0,
            openai_compat_top_p=0.95,
        )
    )
    assert model.temperature == 0.4
    assert model.top_p is None


def test_parametros_de_generacion_para_el_informe() -> None:
    settings = _compat(
        llm_provider="gemini",
        llm_fallback_providers="openai_compat",
        llm_temperature=0.4,
        openai_compat_temperature=1.0,
        openai_compat_top_p=0.95,
    )

    assert generation_settings(settings) == {
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "temperature": 0.4,
        "top_p": None,
        "structured_mode": "native",
        "max_output_tokens": 1024,
        "timeout_seconds": 30,
    }
    compat = generation_settings(settings, "openai_compat")
    assert compat == {
        "provider": "openai_compat",
        "model": "nvidia:deepseek-ai/modelo-falso",
        "temperature": 1.0,
        "top_p": 0.95,
        "structured_mode": "function_calling",
        "max_output_tokens": 4096,
        "timeout_seconds": 60,
    }
    assert FAKE_KEY not in repr(compat)
