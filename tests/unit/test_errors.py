import pytest

from spark_rag.rag.errors import describe_error


def test_incluye_tipo_y_motivo() -> None:
    error = RuntimeError("503 This model is currently experiencing high demand.")
    assert describe_error(error) == (
        "RuntimeError: 503 This model is currently experiencing high demand."
    )


def test_sin_mensaje_solo_el_tipo() -> None:
    assert describe_error(TimeoutError()) == "TimeoutError"


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        ("fallo con key=AIzaSyFAKEFAKEFAKEFAKEFAKEFAKE", "AIzaSyFAKE"),
        ("clave AIzaSyFAKEFAKEFAKEFAKEFAKEFAKE inválida", "AIzaSyFAKE"),
        ("401 nvapi-FAKEFAKEFAKEFAKE rechazada", "nvapi-FAKE"),
        ("Incorrect API key provided: sk-FAKEFAKEFAKEFAKE", "sk-FAKE"),
        ("Authorization: Bearer abc.def.ghi", "abc.def.ghi"),
        ("api_key=supersecreta&x=1", "supersecreta"),
        ("postgresql://spark:secreta@db:5432/spark", "secreta"),
    ],
)
def test_enmascara_credenciales(message: str, secret: str) -> None:
    reason = describe_error(ValueError(message))
    assert secret not in reason
    assert "***" in reason


def test_recorta_mensajes_largos_y_saltos_de_linea() -> None:
    reason = describe_error(RuntimeError("línea\n" * 100))
    assert "\n" not in reason
    assert len(reason) <= 200
