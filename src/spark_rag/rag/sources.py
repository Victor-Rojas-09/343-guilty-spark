"""Fragmentos recuperados, jerarquía de fuentes y bloque de contexto.

El manual (§2.4, RAG-06) exige distinguir el tipo de evidencia. El corpus no
trae sidecars con `canon_status`, así que la clase de autoridad se deriva de la
carpeta (`category`) y, si en el futuro un sidecar aporta `canon_status`, este
tiene prioridad. Nada de esto modifica la ingesta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SourceClass(StrEnum):
    """Clases de autoridad de la evidencia documental."""

    CANON = "canon"
    TRANSCRIPTS = "transcripciones"
    REFERENCE = "referencia"
    COMMUNITY = "comunidad"
    UNCLASSIFIED = "sin_clasificar"


# Orden de prioridad al presentar evidencia al modelo (menor = más autoridad).
AUTHORITY_RANK: dict[SourceClass, int] = {
    SourceClass.CANON: 0,
    SourceClass.TRANSCRIPTS: 1,
    SourceClass.REFERENCE: 2,
    SourceClass.UNCLASSIFIED: 3,
    SourceClass.COMMUNITY: 4,
}

# Descripción que ve el modelo junto a cada documento.
SOURCE_CLASS_LABELS: dict[SourceClass, str] = {
    SourceClass.CANON: (
        "Dossier del proyecto basado en canon oficial (juegos, novelas y "
        "publicaciones de Halo/Xbox)"
    ),
    SourceClass.TRANSCRIPTS: (
        "Guía de diálogos y archivo de transcripciones: localizador de escenas; "
        "contrastar con la obra original"
    ),
    SourceClass.REFERENCE: (
        "Catálogo de fuentes y bibliografía (incluye referencias secundarias "
        "como Halopedia)"
    ),
    SourceClass.COMMUNITY: "Interpretación comunitaria (Reddit/debates): NO es canon",
    SourceClass.UNCLASSIFIED: "Fuente sin clasificar: usar con prudencia",
}

CATEGORY_TO_CLASS: dict[str, SourceClass] = {
    "history": SourceClass.CANON,
    "dialogs": SourceClass.TRANSCRIPTS,
    "sources": SourceClass.REFERENCE,
    "community": SourceClass.COMMUNITY,
}

# Valores de `canon_status` de sidecar reconocidos (si existieran).
CANON_STATUS_TO_CLASS: dict[str, SourceClass] = {
    "canon_primario": SourceClass.CANON,
    "canon": SourceClass.CANON,
    "transcripcion": SourceClass.TRANSCRIPTS,
    "transcripciones": SourceClass.TRANSCRIPTS,
    "referencia_secundaria": SourceClass.REFERENCE,
    "referencia": SourceClass.REFERENCE,
    "interpretacion_comunitaria": SourceClass.COMMUNITY,
    "comunidad": SourceClass.COMMUNITY,
}

NARRATIVE_FIELDS = ("work", "edition", "level", "terminal", "scene", "speaker")


def classify_source(metadata: dict[str, Any]) -> SourceClass:
    """Determina la clase de autoridad de un fragmento a partir de sus metadatos."""
    canon_status = metadata.get("canon_status")
    if isinstance(canon_status, str):
        mapped = CANON_STATUS_TO_CLASS.get(canon_status.strip().lower())
        if mapped is not None:
            return mapped
    category = str(metadata.get("category") or "").strip().lower()
    return CATEGORY_TO_CLASS.get(category, SourceClass.UNCLASSIFIED)


@dataclass(frozen=True)
class RetrievedChunk:
    """Fragmento recuperado de pgvector con su similitud coseno (0..1)."""

    content: str
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source") or "desconocido")

    @property
    def category(self) -> str:
        return str(self.metadata.get("category") or "general")

    @property
    def page(self) -> int | None:
        page = self.metadata.get("page")
        try:
            return int(page) if page is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def source_class(self) -> SourceClass:
        return classify_source(self.metadata)

    def narrative_locators(self) -> dict[str, str]:
        """Campos narrativos presentes en los metadatos (nunca inventados)."""
        locators: dict[str, str] = {}
        for name in NARRATIVE_FIELDS:
            value = self.metadata.get(name)
            if value is None or isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                locators[name] = text
        return locators


def sort_by_authority(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Ordena por autoridad y, dentro de cada clase, por similitud descendente."""
    return sorted(
        chunks,
        key=lambda chunk: (AUTHORITY_RANK[chunk.source_class], -chunk.similarity),
    )


def public_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Nombres de fuente únicos, en el orden de autoridad del contexto."""
    names: list[str] = []
    for chunk in sort_by_authority(chunks):
        if chunk.source not in names:
            names.append(chunk.source)
    return names


def source_details(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Detalle por fuente y página (deduplicado), con la mejor similitud.

    Devuelve diccionarios planos para no acoplar el motor RAG a la API; la capa
    HTTP los valida con `SourceDetail`.
    """
    best: dict[tuple[str, int | None], RetrievedChunk] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        current = best.get(key)
        if current is None or chunk.similarity > current.similarity:
            best[key] = chunk

    return [
        {
            "source": chunk.source,
            "category": chunk.category,
            "page": chunk.page,
            "source_class": chunk.source_class.value,
            "similarity": round(chunk.similarity, 4),
            **chunk.narrative_locators(),
        }
        for chunk in sort_by_authority(list(best.values()))
    ]


def _escape(text: str) -> str:
    """Neutraliza etiquetas `<documento>` y comillas dentro de la evidencia."""
    return text.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Construye el bloque de evidencia delimitado que se entrega al modelo.

    Cada fragmento va en su propio `<documento>` con su procedencia. El texto se
    escapa para que no pueda simular el cierre de un documento ni abrir
    instrucciones nuevas.
    """
    blocks: list[str] = []
    for index, chunk in enumerate(sort_by_authority(chunks), start=1):
        attributes = [
            f'id="{index}"',
            f'source="{_escape(chunk.source)}"',
            f'categoria="{_escape(chunk.category)}"',
            f'clase="{chunk.source_class.value}"',
        ]
        if chunk.page is not None:
            attributes.append(f'pagina="{chunk.page}"')
        for name, value in chunk.narrative_locators().items():
            attributes.append(f'{name}="{_escape(value)}"')
        label = SOURCE_CLASS_LABELS[chunk.source_class]
        blocks.append(
            f"<documento {' '.join(attributes)}>\n"
            f"[{label}]\n"
            f"{_escape(chunk.content.strip())}\n"
            "</documento>"
        )
    return "\n\n".join(blocks)
