"""Motor RAG de la Etapa 2.

* `retriever`: recuperación semántica en pgvector (top-k, umbral, filtros).
* `sources`: jerarquía de fuentes y bloque de evidencia delimitado.
* `prompts`: System Prompt versionado de 343 Guilty Spark.
* `llm`: fábrica del modelo de chat configurable.
* `chain`: cadena LangChain con salida estructurada `{reply, mood}`.
* `history` / `service`: sesiones, límite de historial y orquestación.
"""
