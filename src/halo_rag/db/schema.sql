-- Esquema inicial propuesto para el sistema RAG.
-- Se aplicará desde src/halo_rag/db/bootstrap.py en la Fase 2.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    source_path TEXT NOT NULL,
    checksum    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    token_count INTEGER,
    embedding   vector(768),  -- Debe coincidir con EMBEDDING_DIM.
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Índice de similitud: decisión pendiente (HNSW vs IVFFlat y métrica).
-- Ejemplo a evaluar:
-- CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
--     ON chunks USING hnsw (embedding vector_cosine_ops);
