from spark_rag.rag.history import ChatTurn, trim_history


def _conversation(pairs: int) -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    for index in range(pairs):
        turns.append(ChatTurn(role="user", content=f"pregunta {index}"))
        turns.append(ChatTurn(role="assistant", content=f"respuesta {index}"))
    return turns


def test_limita_numero_de_mensajes() -> None:
    trimmed = trim_history(_conversation(10), max_messages=4, max_chars=0)

    assert [t.content for t in trimmed] == [
        "pregunta 8",
        "respuesta 8",
        "pregunta 9",
        "respuesta 9",
    ]


def test_no_empieza_con_respuesta_huerfana() -> None:
    trimmed = trim_history(_conversation(3), max_messages=3, max_chars=0)

    assert trimmed[0].role == "user"
    assert [t.content for t in trimmed] == ["pregunta 2", "respuesta 2"]


def test_limita_caracteres_descartando_lo_mas_antiguo() -> None:
    turns = [
        ChatTurn(role="user", content="a" * 50),
        ChatTurn(role="assistant", content="b" * 50),
        ChatTurn(role="user", content="c" * 30),
        ChatTurn(role="assistant", content="d" * 30),
    ]
    trimmed = trim_history(turns, max_messages=10, max_chars=100)

    assert [t.content[0] for t in trimmed] == ["c", "d"]


def test_mensaje_unico_demasiado_largo_se_descarta() -> None:
    turns = [ChatTurn(role="user", content="x" * 500)]
    assert trim_history(turns, max_messages=10, max_chars=100) == []


def test_limite_cero_desactiva_historial() -> None:
    assert trim_history(_conversation(2), max_messages=0, max_chars=1000) == []


def test_historial_vacio() -> None:
    assert trim_history([], max_messages=5, max_chars=100) == []
