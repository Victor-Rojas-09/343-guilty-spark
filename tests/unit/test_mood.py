import pytest

from spark_rag.rag.mood import DEFAULT_MOOD, Mood, normalize_mood


def test_conjunto_controlado() -> None:
    assert {mood.value for mood in Mood} == {
        "idle",
        "thinking",
        "enthusiastic",
        "rampant",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("thinking", Mood.THINKING),
        (" Enthusiastic ", Mood.ENTHUSIASTIC),
        ("RAMPANT", Mood.RAMPANT),
        (Mood.IDLE, Mood.IDLE),
    ],
)
def test_normaliza_valores_validos(value: object, expected: Mood) -> None:
    assert normalize_mood(value) is expected


@pytest.mark.parametrize("value", ["feliz", "", None, 3, ["idle"]])
def test_valores_invalidos_usan_el_mood_por_defecto(value: object) -> None:
    assert normalize_mood(value) is DEFAULT_MOOD
    assert DEFAULT_MOOD is Mood.THINKING
