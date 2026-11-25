import hashlib
import json
from pathlib import Path

import pytest

from spark_rag.ingestion.metadata import (
    build_document_metadata,
    compute_sha256,
    infer_category,
    make_chunk_id,
    relative_source,
)


def _crear_archivo(base: Path, relative: str) -> Path:
    path = base / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"contenido")
    return path


def test_metadatos_minimos_de_pdf(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = _crear_archivo(data_dir, "history/01_Dossier.pdf")

    metadata = build_document_metadata(path, data_dir, page=0)

    assert metadata == {
        "source": "history/01_Dossier.pdf",
        "category": "history",
        "page": 1,
        "source_type": "pdf",
    }


def test_pagina_logica_para_texto(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = _crear_archivo(data_dir, "notes.txt")

    metadata = build_document_metadata(path, data_dir, page=None)

    assert metadata["source"] == "notes.txt"
    assert metadata["category"] == "general"
    assert metadata["page"] == 1
    assert metadata["source_type"] == "text"


def test_sidecar_solo_anade_campos_opcionales(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = _crear_archivo(data_dir, "dialogs/02_Guia.pdf")
    sidecar = path.with_suffix(".pdf.metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "work": "Halo 3",
                "terminal": "06",
                "source": "NO_DEBE_SOBRESCRIBIR",
                "reliability": "alta",
            }
        ),
        encoding="utf-8",
    )

    metadata = build_document_metadata(path, data_dir, page=4)

    assert metadata["source"] == "dialogs/02_Guia.pdf"
    assert metadata["page"] == 5
    assert metadata["work"] == "Halo 3"
    assert metadata["terminal"] == "06"
    assert metadata["reliability"] == "alta"


def test_sha256_conocido(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"abc")
    assert compute_sha256(path) == hashlib.sha256(b"abc").hexdigest()


def test_source_fuera_del_corpus(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        relative_source(outside, data_dir)


def test_categoria_de_archivo_raiz() -> None:
    assert infer_category("archivo.txt") == "general"
    assert infer_category("sources/03_Fuentes.pdf") == "sources"


def test_chunk_id_estable_y_unico() -> None:
    first = make_chunk_id("history/01.pdf", 1, 0)
    same = make_chunk_id("history/01.pdf", 1, 0)
    different = make_chunk_id("history/01.pdf", 1, 1)

    assert first == same
    assert first != different
