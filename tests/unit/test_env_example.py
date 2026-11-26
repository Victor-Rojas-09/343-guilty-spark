"""`.env.example` y `docker-compose.yml` no deben llevar secretos reales."""

import re
from pathlib import Path

from spark_rag.config import PROJECT_ROOT, Settings

ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
COMPOSE = PROJECT_ROOT / "docker-compose.yml"
SECRET_NAME = re.compile(r"(API_KEY|TOKEN|SECRET)$")
SECRET_VALUE = re.compile(
    r"AIza[0-9A-Za-z_\-]{20,}|nvapi-[0-9A-Za-z_\-]{8,}|\bsk-[0-9A-Za-z_\-]{8,}"
)


def _entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            entries[name.strip()] = value.strip()
    return entries


def test_variables_secretas_van_vacias() -> None:
    entries = _entries(ENV_EXAMPLE)
    secrets = {
        name: value for name, value in entries.items() if SECRET_NAME.search(name)
    }

    assert {"GEMINI_API_KEY", "OPENAI_COMPAT_API_KEY"} <= set(secrets)
    assert all(value == "" for value in secrets.values()), secrets


def test_no_hay_valores_con_forma_de_clave() -> None:
    assert not SECRET_VALUE.search(ENV_EXAMPLE.read_text(encoding="utf-8"))
    assert not SECRET_VALUE.search(COMPOSE.read_text(encoding="utf-8"))


def test_password_de_postgres_es_el_de_desarrollo() -> None:
    entries = _entries(ENV_EXAMPLE)
    assert entries["POSTGRES_PASSWORD"] == Settings(_env_file=None).postgres_password


def test_todas_las_variables_de_settings_estan_documentadas() -> None:
    documented = set(_entries(ENV_EXAMPLE))
    expected = {name.upper() for name in Settings.model_fields}
    assert expected <= documented, expected - documented


def test_compose_pasa_las_variables_de_respaldo_sin_exigir_clave() -> None:
    compose = COMPOSE.read_text(encoding="utf-8")

    assert "OPENAI_COMPAT_API_KEY: ${OPENAI_COMPAT_API_KEY:-}" in compose
    for name in (
        "LLM_FALLBACK_PROVIDERS",
        "LLM_FALLBACK_TIMEOUT_SECONDS",
        "LLM_PRIMARY_COOLDOWN_SECONDS",
        "OPENAI_COMPAT_BASE_URL",
        "OPENAI_COMPAT_CHAT_MODEL",
        "OPENAI_COMPAT_LABEL",
        "OPENAI_COMPAT_STRUCTURED_MODE",
        "OPENAI_COMPAT_TIMEOUT_SECONDS",
        "OPENAI_COMPAT_MAX_OUTPUT_TOKENS",
        "OPENAI_COMPAT_EXTRA_BODY",
        "OPENAI_COMPAT_TEMPERATURE",
        "OPENAI_COMPAT_TOP_P",
    ):
        assert f"{name}: ${{{name}:-" in compose, name
