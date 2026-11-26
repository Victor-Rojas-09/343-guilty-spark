import logging
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from spark_rag.config import Settings
from spark_rag.rag.chain import SparkRagChain
from spark_rag.rag.errors import (
    EmptyModelResponse,
    LLMConfigurationError,
    LLMUnavailable,
)
from spark_rag.rag.fallback import (
    Attempt,
    Candidate,
    FallbackGenerator,
    build_fallback_generator,
)
from spark_rag.rag.mood import Mood
from spark_rag.rag.sources import RetrievedChunk
from spark_rag.rag.structured import TOOL_NAME, SparkOutput

FAKE_KEY = "nvapi-FAKEFAKEFAKEFAKE"
MESSAGES = [HumanMessage(content="¿Quién eres?")]
CHUNK = RetrievedChunk(
    content="Chakas fue un humano de Erde-Tyrene.",
    similarity=0.8,
    metadata={"source": "history/dossier.pdf", "category": "history", "page": 3},
)


class FakeAPIError(Exception):
    """Error HTTP con `status_code`, como los de `openai`."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class _Model:
    """Generador falso: devuelve una salida o lanza un error, y cuenta llamadas."""

    def __init__(self, output: Any) -> None:
        self.output = output
        self.calls = 0

    def __call__(self, _messages: Any) -> Any:
        self.calls += 1
        if isinstance(self.output, BaseException):
            raise self.output
        return self.output


class _Clock:
    """Reloj falso: cada lectura avanza `step` segundos."""

    def __init__(self, step: float = 0.5) -> None:
        self.now = 100.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _ok(reply: str = "Saludos, Reclamador.") -> SparkOutput:
    return SparkOutput(reply=reply, mood="enthusiastic")


def _generator(*outputs: Any, clock: _Clock | None = None):
    models = [_Model(output) for output in outputs]
    names = ["gemini-3.8-flash", "nvidia:mistralai/modelo", "tercero"]
    candidates = [
        Candidate(model=names[i], generator=RunnableLambda(m), structured_mode=f"m{i}")
        for i, m in enumerate(models)
    ]
    return FallbackGenerator(candidates, clock=clock or _Clock()), models


# --- Qué activa el respaldo --------------------------------------------------


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (FakeAPIError(429, "Resource exhausted: quota"), "rate_limit"),
        (
            FakeAPIError(503, "This model is currently experiencing high demand"),
            "unavailable",
        ),
        (TimeoutError("read timed out"), "timeout"),
        (ConnectionError("connection reset"), "connection"),
        (EmptyModelResponse("sin texto"), "empty_response"),
    ],
)
def test_fallo_de_disponibilidad_pasa_al_respaldo(
    error: Exception, outcome: str
) -> None:
    generator, (primary, fallback) = _generator(error, _ok())

    generation = generator.generate(MESSAGES)

    assert (primary.calls, fallback.calls) == (1, 1)
    assert generation.reply == "Saludos, Reclamador."
    assert generation.mood is Mood.ENTHUSIASTIC
    assert generation.model == "nvidia:mistralai/modelo"
    assert generation.structured_mode == "m1"
    assert [(a.model, a.outcome) for a in generation.attempts] == [
        ("gemini-3.8-flash", outcome),
        ("nvidia:mistralai/modelo", "ok"),
    ]


def test_respuesta_vacia_tras_parsear_pasa_al_respaldo() -> None:
    generator, _ = _generator(_ok("   "), _ok())

    generation = generator.generate(MESSAGES)

    assert generation.model == "nvidia:mistralai/modelo"
    assert generation.attempts[0].outcome == "empty_response"


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_error_de_configuracion_no_activa_el_respaldo(status: int) -> None:
    generator, (primary, fallback) = _generator(FakeAPIError(status, "invalid"), _ok())

    with pytest.raises(LLMUnavailable) as info:
        generator.generate(MESSAGES)

    assert (primary.calls, fallback.calls) == (1, 0)
    assert [a.outcome for a in info.value.attempts] == ["configuration"]


def test_error_desconocido_no_activa_el_respaldo() -> None:
    generator, (_, fallback) = _generator(RuntimeError("bug"), _ok())

    with pytest.raises(LLMUnavailable) as info:
        generator.generate(MESSAGES)

    assert fallback.calls == 0
    assert [a.outcome for a in info.value.attempts] == ["other"]


def test_todos_fallan_es_llm_unavailable_con_los_intentos() -> None:
    generator, models = _generator(
        FakeAPIError(503, "high demand"),
        TimeoutError("read timed out"),
        FakeAPIError(429, "rate limit"),
    )

    with pytest.raises(LLMUnavailable) as info:
        generator.generate(MESSAGES)

    assert [m.calls for m in models] == [1, 1, 1]
    assert [a.outcome for a in info.value.attempts] == [
        "unavailable",
        "timeout",
        "rate_limit",
    ]
    assert isinstance(info.value.__cause__, FakeAPIError)


def test_el_principal_que_responde_no_llama_al_respaldo() -> None:
    generator, (primary, fallback) = _generator(_ok(), _ok("otra"))

    generation = generator.generate(MESSAGES)

    assert (primary.calls, fallback.calls) == (1, 0)
    assert generation.model == "gemini-3.8-flash"
    assert [a.outcome for a in generation.attempts] == ["ok"]


def test_un_solo_proveedor_falla_como_antes() -> None:
    generator, _ = _generator(FakeAPIError(503, "high demand"))

    with pytest.raises(LLMUnavailable) as info:
        generator.generate(MESSAGES)
    assert [a.outcome for a in info.value.attempts] == ["unavailable"]


def test_registra_la_duracion_de_cada_intento() -> None:
    generator, _ = _generator(TimeoutError("x"), _ok(), clock=_Clock(step=1.25))

    generation = generator.generate(MESSAGES)

    assert [a.seconds for a in generation.attempts] == [1.25, 1.25]
    assert generation.attempts[0].as_dict() == {
        "model": "gemini-3.8-flash",
        "outcome": "timeout",
        "seconds": 1.25,
    }


def test_attempt_redondea_a_milisegundos() -> None:
    assert Attempt("m", "ok", 1.23456).as_dict()["seconds"] == 1.235


def test_cadena_vacia_no_se_permite() -> None:
    with pytest.raises(ValueError, match="al menos un proveedor"):
        FallbackGenerator([])


# --- Logs sin claves ------------------------------------------------------------


def test_logs_y_errores_sin_claves(caplog: pytest.LogCaptureFixture) -> None:
    generator, _ = _generator(
        FakeAPIError(429, f"quota exceeded Authorization: Bearer {FAKE_KEY}"),
        FakeAPIError(503, f"high demand api_key={FAKE_KEY}"),
    )

    with (
        caplog.at_level(logging.INFO, logger="spark_rag.rag.fallback"),
        pytest.raises(LLMUnavailable) as info,
    ):
        generator.generate(MESSAGES)

    assert "quota exceeded" in caplog.text
    assert "Se prueba nvidia:mistralai/modelo" in caplog.text
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in str(info.value)
    assert FAKE_KEY not in repr(info.value.attempts)


def test_log_del_respaldo_que_responde(caplog: pytest.LogCaptureFixture) -> None:
    generator, _ = _generator(FakeAPIError(503, "high demand"), _ok())

    with caplog.at_level(logging.INFO, logger="spark_rag.rag.fallback"):
        generator.generate(MESSAGES)

    assert "Respondió el respaldo nvidia:mistralai/modelo" in caplog.text


# --- Construcción desde la configuración ---------------------------------------


def _fake_llm(reply: str) -> SimpleNamespace:
    """Modelo falso con las dos formas de salida estructurada."""

    def with_structured_output(schema: Any, include_raw: bool) -> RunnableLambda:
        assert (schema, include_raw) == (SparkOutput, True)
        return RunnableLambda(
            lambda _: {
                "raw": AIMessage(content=reply),
                "parsed": None,
                "parsing_error": None,
            }
        )

    def bind_tools(tools: list[Any], tool_choice: str) -> RunnableLambda:
        assert (tools, tool_choice) == ([SparkOutput], TOOL_NAME)
        return RunnableLambda(
            lambda _: AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": TOOL_NAME,
                        "args": {"reply": reply, "mood": "idle"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        )

    return SimpleNamespace(
        with_structured_output=with_structured_output, bind_tools=bind_tools
    )


def _settings(**kwargs: object) -> Settings:
    base: dict[str, object] = {
        "gemini_api_key": "fake",
        "openai_compat_api_key": FAKE_KEY,
        "openai_compat_chat_model": "mistralai/modelo",
    }
    return Settings(_env_file=None, **{**base, **kwargs})


def test_construye_principal_y_respaldo_en_orden() -> None:
    built: list[str] = []

    def factory(settings: Settings, provider: str) -> SimpleNamespace:
        built.append(provider)
        return _fake_llm(f"respuesta de {provider}")

    generator = build_fallback_generator(
        _settings(llm_fallback_providers="openai_compat"), model_factory=factory
    )

    assert built == ["gemini", "openai_compat"]
    assert generator.models == ["gemini-3.8-flash", "nvidia:mistralai/modelo"]
    generation = generator.generate(MESSAGES)
    assert generation.reply == "respuesta de gemini"
    assert generation.structured_mode == "native"


def test_respaldo_usa_el_modo_estructurado_configurado() -> None:
    def factory(settings: Settings, provider: str) -> Any:
        if provider == "gemini":
            return SimpleNamespace(
                with_structured_output=lambda *a, **k: RunnableLambda(
                    _Model(FakeAPIError(503, "high demand"))
                )
            )
        return _fake_llm("respuesta del respaldo")

    generator = build_fallback_generator(
        _settings(llm_fallback_providers="openai_compat"), model_factory=factory
    )
    generation = generator.generate(MESSAGES)

    assert generation.model == "nvidia:mistralai/modelo"
    assert generation.structured_mode == "function_calling"
    assert (generation.reply, generation.mood, generation.structured) == (
        "respuesta del respaldo",
        Mood.IDLE,
        True,
    )


def test_sin_respaldo_hay_un_solo_proveedor() -> None:
    generator = build_fallback_generator(
        _settings(), model_factory=lambda s, p: _fake_llm("x")
    )
    assert generator.models == ["gemini-3.8-flash"]


def test_openai_compat_como_principal_no_construye_gemini() -> None:
    built: list[str] = []

    def factory(settings: Settings, provider: str) -> SimpleNamespace:
        built.append(provider)
        return _fake_llm("x")

    generator = build_fallback_generator(
        _settings(llm_provider="openai_compat"), model_factory=factory
    )
    assert built == ["openai_compat"]
    assert generator.models == ["nvidia:mistralai/modelo"]


def test_respaldo_sin_clave_falla_al_arrancar() -> None:
    # Modelos reales, sin red: construirlos no hace ninguna petición.
    settings = _settings(
        llm_fallback_providers="openai_compat", openai_compat_api_key=""
    )
    with pytest.raises(LLMConfigurationError, match="OPENAI_COMPAT_API_KEY"):
        build_fallback_generator(settings)


def test_construye_modelos_reales_sin_red() -> None:
    generator = build_fallback_generator(
        _settings(llm_fallback_providers="openai_compat")
    )
    assert generator.models == ["gemini-3.8-flash", "nvidia:mistralai/modelo"]


# --- Integración con la cadena ---------------------------------------------------


def _chain(*outputs: Any) -> SparkRagChain:
    generator, _ = _generator(*outputs)
    return SparkRagChain(retrieve=lambda _: [CHUNK], generator=generator)


def test_cadena_indica_el_modelo_y_los_intentos() -> None:
    result = _chain(FakeAPIError(429, "quota"), _ok()).invoke("¿Quién eres?")

    assert result.reply == "Saludos, Reclamador."
    assert result.model == "nvidia:mistralai/modelo"
    assert result.structured_mode == "m1"
    assert result.fallback_used is True
    assert [a.outcome for a in result.attempts] == ["rate_limit", "ok"]
    assert result.sources == ["history/dossier.pdf"]


def test_cadena_sin_respaldo_usado() -> None:
    result = _chain(_ok()).invoke("¿Quién eres?")
    assert result.model == "gemini-3.8-flash"
    assert result.fallback_used is False


def test_cadena_con_generador_simple_usa_model_name() -> None:
    chain = SparkRagChain(
        retrieve=lambda _: [CHUNK],
        generator=RunnableLambda(lambda _: _ok()),
        model_name="gemini-3.8-flash",
    )
    result = chain.invoke("x")
    assert result.model == "gemini-3.8-flash"
    assert [a.outcome for a in result.attempts] == ["ok"]


def test_cadena_todos_fallan_conserva_intentos_sin_claves(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chain = _chain(
        FakeAPIError(503, f"high demand key={FAKE_KEY}"),
        FakeAPIError(503, f"overloaded Bearer {FAKE_KEY}"),
    )
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(LLMUnavailable) as info,
    ):
        chain.invoke("x")

    assert [a.outcome for a in info.value.attempts] == ["unavailable", "unavailable"]
    assert (
        "overloaded" in caplog.text
    )  # motivo del último fallo, en el log de la cadena
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in str(info.value)
