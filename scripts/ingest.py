"""Punto de entrada CLI para ingestar el manual en la base vectorial."""

from halo_rag.config import get_settings


def main() -> None:
    """Ejecuta la ingesta del PDF configurado (Fase 2)."""
    settings = get_settings()
    raise SystemExit(
        "Ingesta pendiente de implementación (Fase 2). "
        f"PDF configurado: {settings.pdf_path}"
    )


if __name__ == "__main__":
    main()
