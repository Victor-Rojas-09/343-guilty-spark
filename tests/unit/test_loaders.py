from pathlib import Path

from spark_rag.ingestion.loaders import iter_source_files, load_file


def test_descubre_formatos_soportados_recursivamente(tmp_path: Path) -> None:
    files = {
        "a.pdf": b"pdf",
        "history/b.md": b"md",
        "history/c.markdown": b"markdown",
        "sources/d.txt": b"txt",
        "sources/ignored.json": b"{}",
        ".hidden.pdf": b"hidden",
        ".hidden/inside.txt": b"hidden",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    discovered = iter_source_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in discovered] == [
        "a.pdf",
        "history/b.md",
        "history/c.markdown",
        "sources/d.txt",
    ]


def test_carga_markdown_y_texto(tmp_path: Path) -> None:
    markdown = tmp_path / "nota.md"
    markdown.write_text("# Título\n\nContenido", encoding="utf-8")
    text = tmp_path / "nota.txt"
    text.write_text("Contenido de prueba", encoding="utf-8")

    markdown_docs = load_file(markdown)
    text_docs = load_file(text)

    assert "# Título" in markdown_docs[0].page_content
    assert text_docs[0].page_content.strip() == "Contenido de prueba"


def test_carga_pdf_por_paginas() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    pdf = next(data_dir.rglob("*.pdf"))

    documents = load_file(pdf)

    assert documents
    assert all("page" in document.metadata for document in documents)
