-- Se ejecuta automáticamente la primera vez que se inicializa el volumen.
-- Habilita pgvector para almacenar embeddings y consultar por similitud.
CREATE EXTENSION IF NOT EXISTS vector;

-- Falla de forma explícita si la extensión no quedó disponible.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
    ) THEN
        RAISE EXCEPTION 'La extensión pgvector no pudo habilitarse.';
    END IF;
END
$$;
