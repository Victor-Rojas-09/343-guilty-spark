-- Se ejecuta automáticamente la primera vez que se inicializa el volumen de datos.
-- Habilita pgvector para almacenar embeddings y realizar búsquedas por similitud.
CREATE EXTENSION IF NOT EXISTS vector;
