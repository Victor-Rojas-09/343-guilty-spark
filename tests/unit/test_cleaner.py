from spark_rag.ingestion.cleaner import clean_text


def test_normaliza_espacios_y_saltos_de_linea() -> None:
    raw = "  Hola   mundo \r\n\r\n\r\n  Adiós\tmundo  "
    assert clean_text(raw) == "Hola mundo\n\nAdiós mundo"


def test_recompone_palabras_cortadas_con_guion() -> None:
    assert clean_text("pro-\ngrama") == "programa"


def test_elimina_caracteres_de_control_y_ancho_cero() -> None:
    assert clean_text("a\x00b\u200bc") == "abc"


def test_texto_vacio() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""
