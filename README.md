# 343 Guilty Spark // Asistente conversacional con RAG

Backend del asistente basado en el corpus documental de **343 Guilty Spark**.
Esta entrega implementa únicamente la **Etapa 1: infraestructura e ingesta
documental**:

- PostgreSQL 16 + `pgvector` reproducible con Docker Compose.
- Configuración centralizada y secretos fuera del repositorio.
- Ingesta recursiva de PDF, Markdown y TXT con LangChain.
- Limpieza, chunking configurable y metadatos de procedencia.
- Embeddings de Gemini (`gemini-embedding-001`) validados antes de persistir.
- Persistencia vectorial con `langchain-postgres`.
- Control SHA256 por archivo para evitar duplicados y reindexar cambios.
- API FastAPI mínima con `GET /health`.

El chat, la personalidad, las sesiones y el frontend quedan para las etapas
siguientes. `src/spark_rag/rag/` y `src/spark_rag/tts/` se mantienen como
placeholders aislados.

---

## Arquitectura

```text
.
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── docker/
│   └── init/
│       └── 01-extensions.sql
├── data/                     # Corpus actual (PDF, MD y TXT)
│   ├── community/
│   ├── dialogs/
│   ├── history/
│   └── sources/
├── docs/                     # Manual de requisitos y documentos de apoyo
├── frontend/                 # Placeholder para la Etapa 3
├── scripts/
│   ├── ingest.py
│   └── test_search.py
├── src/
│   └── spark_rag/
│       ├── __init__.py
│       ├── config.py
│       ├── db/
│       │   ├── __init__.py
│       │   ├── connection.py
│       │   └── models.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   ├── cleaner.py
│       │   ├── metadata.py
│       │   ├── embeddings.py
│       │   └── pipeline.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── main.py
│       ├── rag/
│       │   └── __init__.py
│       └── tts/
│           └── __init__.py
└── tests/
    ├── unit/
    └── integration/
```

### Responsabilidades

| Módulo | Responsabilidad |
| --- | --- |
| `config.py` | Configuración tipada desde `.env`/entorno y URLs de conexión. |
| `db/connection.py` | Conexión psycopg, motor SQLAlchemy y health check de pgvector. |
| `db/models.py` | Esquema de control, hashes, runs y borrado de vectores por `source`. |
| `ingestion/loaders.py` | Descubrimiento recursivo y loaders de LangChain por formato. |
| `ingestion/cleaner.py` | Normalización de texto antes del chunking. |
| `ingestion/metadata.py` | `source`, `category`, `page`, `source_type`, SHA256 e IDs estables. |
| `ingestion/embeddings.py` | Embeddings Gemini con validación de dimensión y finitud. |
| `ingestion/pipeline.py` | Orquestación idempotente y reporte por archivo. |
| `api/main.py` | `GET /health` verificando PostgreSQL y pgvector. |

---

## Requisitos

- Docker y Docker Compose (plugin `docker compose`).
- Python **3.11 o 3.12** en la imagen Docker, conforme al manual.
- Una API key de Google Gemini para generar embeddings.
- Para ejecución local, el entorno virtual que gestiones desde PyCharm. El
  proyecto declara sus dependencias en `requirements.txt`; no crea ni modifica
  entornos automáticamente.

> El manual recomienda 3.11/3.12 en el contenedor por compatibilidad de ruedas.
> El Dockerfile usa `python:3.12-slim`. Si usas un entorno local 3.14, instala
> las mismas dependencias declaradas y verifica que `psycopg[binary]` y
> `pydantic-core` dispongan de rueda compatible.

---

## Configuración

Copia la plantilla y completa la clave de Gemini:

```bash
cp .env.example .env
```

Variables principales:

| Variable | Uso |
| --- | --- |
| `POSTGRES_*` | Credenciales y host de PostgreSQL. |
| `GEMINI_API_KEY` | Clave privada de Google AI Studio. No va al frontend. |
| `GEMINI_EMBEDDING_MODEL` | Modelo de embeddings; por defecto `gemini-embedding-001`. |
| `EMBEDDING_DIM` | Dimensión solicitada al modelo; debe coincidir con la columna vectorial. |
| `DATA_DIR` | Raíz del corpus; por defecto `data`. |
| `COLLECTION_NAME` | Colección de `langchain-postgres`. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Tamaño y solape del chunking. |
| `LOG_LEVEL` | Nivel de logging. |

`.env` está ignorado por git. El repositorio solo incluye valores de ejemplo no
secretos en `.env.example`.

---

## Arranque

```bash
docker compose up -d --build
```

Servicios:

- `spark-rag-postgres`: PostgreSQL 16 con la extensión `vector`, volumen
  persistente `pgvector_data` y health check.
- `spark-rag-api`: FastAPI en `http://localhost:8000`, con el corpus montado en
  `/app/data` en modo lectura.

Verificación:

```bash
# Estado de servicios (postgres y api deben quedar healthy)
docker compose ps

# Versión del servidor
docker compose exec postgres sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT version();"'

# Extensión pgvector (debe aparecer "vector")
docker compose exec postgres sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dx"'

# Health de la API
curl -s http://localhost:8000/health
```

Respuesta esperada con todo operativo:

```json
{
  "status": "ok",
  "service": "spark-rag",
  "version": "0.1.0",
  "database": true,
  "pgvector": true,
  "pgvector_version": "0.8.x"
}
```

Si PostgreSQL no responde o falta `vector`, la API devuelve HTTP `503` y
`status: "degraded"` sin exponer credenciales ni trazas internas.

---

## Ingesta

### Dentro de Docker

```bash
docker compose run --rm api python scripts/ingest.py
```

Opciones útiles:

```bash
docker compose run --rm api python scripts/ingest.py \
  --data-dir /app/data \
  --collection spark_documents \
  --log-level INFO
```

### En el entorno local gestionado por PyCharm

Con las dependencias de `requirements.txt` ya instaladas en tu entorno:

```bash
python scripts/ingest.py
```

El script añade `src/` al `PYTHONPATH` automáticamente. También puedes usar un
directorio alternativo con `--data-dir`.

### Qué hace la ingesta

1. Descubre recursivamente archivos `.pdf`, `.md`, `.markdown` y `.txt`.
2. Calcula el SHA256 de cada archivo.
3. Si ya está indexado con el mismo hash, lo omite.
4. Si es nuevo o cambió, carga el documento, limpia el texto y lo divide en
   chunks con `RecursiveCharacterTextSplitter`.
5. Añade metadatos de procedencia a cada chunk.
6. Elimina los vectores anteriores de ese `source` y genera/inserta los nuevos.
7. Registra el resultado por archivo en `ingested_files` y el resumen en
   `ingestion_runs`.
8. Si un archivo está vacío, dañado o no tiene texto extraíble, registra el
   error y continúa con el resto del corpus.

El proceso es idempotente: ejecutar dos veces sin cambios produce
`omitidos=N`, no duplica chunks y no vuelve a llamar a Gemini para esos
archivos. Si un documento cambia, se borran sus fragmentos obsoletos antes de
insertar los nuevos.

### Reporte

La CLI imprime un resumen y una línea por archivo:

```text
Descubiertos=7 indexados=7 omitidos=0 fallidos=0 estado=success
[indexed] history/01_Dossier_de_lore_343_Guilty_Spark.pdf (42 chunks) 12.40s
[indexed] dialogs/02_Guia_de_dialogos_y_transcripciones.pdf (18 chunks) 5.21s
```

Además queda persistido en PostgreSQL:

- `ingested_files`: estado, hash, tamaño, chunks, error y fechas por archivo.
- `ingestion_runs`: inicio, fin, estado y contadores de cada ejecución.

---

## Metadatos

Campos mínimos en cada fragmento:

| Campo | Origen |
| --- | --- |
| `source` | Ruta relativa POSIX dentro de `data/`, por ejemplo `history/01_Dossier.pdf`. |
| `category` | Primera carpeta del corpus (`history`, `dialogs`, `sources`, `community`) o `general`. |
| `page` | Página 1-indexada en PDF; página lógica 1 en MD/TXT sin paginación. |

También se guarda `source_type` (`pdf`, `markdown`, `text`), `chunk_index`,
`chunk_id`, `chunk_size` implícito por configuración y, si existen, los campos
narrativos opcionales del sidecar.

Los campos opcionales **no se inventan**. Si se dispone de información como
obra, edición, nivel, terminal, escena, hablante, estado canónico o fiabilidad,
puede añadirse junto al documento en un sidecar:

```text
data/dialogs/02_Guia_de_dialogos_y_transcripciones.pdf.metadata.json
```

```json
{
  "work": "Halo 3",
  "edition": "Original",
  "level": "The Covenant",
  "terminal": "06",
  "scene": "Interrogatorio",
  "speaker": "343 Guilty Spark",
  "canon_status": "canon_primario",
  "reliability": "alta"
}
```

Solo se aceptan los campos opcionales conocidos; `source`, `category` y `page`
siempre se calculan desde el archivo y no pueden sobrescribirse con el sidecar.

---

## Búsqueda de humo en pgvector

Con el corpus ya ingerido:

```bash
docker compose run --rm api python scripts/test_search.py \
  "Chakas y 343 Guilty Spark" --top-k 5

# Filtrar por categoría
docker compose run --rm api python scripts/test_search.py \
  "Protocolo de activación" --category dialogs
```

El comando devuelve similitud, `source`, `category`, `page` y un extracto del
fragmento. Es una verificación directa de que la colección vectorial contiene
registros trazables.

---

## Pruebas

```bash
# Pruebas unitarias (no requieren base de datos ni Gemini)
pytest tests/unit

# Pruebas de integración con PostgreSQL/pgvector real
RUN_INTEGRATION=1 pytest tests/integration
```

Las pruebas de integración se omiten si `RUN_INTEGRATION` no vale `1`. No se
hacen llamadas reales a Gemini en las pruebas unitarias.

Cobertura actual de la Etapa 1:

- Limpieza y normalización de texto.
- Construcción de metadatos y sidecars.
- Hash SHA256 e IDs estables de chunks.
- Descubrimiento recursivo de formatos.
- Validación de vectores (dimensión, vacío y valores no finitos).
- Configuración y URLs de conexión.
- Disponibilidad de PostgreSQL/pgvector y esquema idempotente.

---

## Decisiones de esta etapa

- **Dimensión de embeddings:** `EMBEDDING_DIM=768` por defecto. El modelo
  `gemini-embedding-001` entrega 3072 dimensiones si no se solicita reducción;
  aquí se solicita explícitamente la dimensión configurada y se valida antes de
  insertar. La columna vectorial se crea con la misma dimensión.
- **Distancia:** `pgvector` se usa con la métrica coseno, coherente con el
  índice HNSW creado (`vector_cosine_ops`).
- **Idempotencia:** el hash SHA256 vive en `ingested_files`. Los IDs de chunk son
  deterministas (`uuid5`), pero la reindexación elimina primero por `source` para
  no conservar fragmentos obsoletos si el documento se acorta.
- **Errores por archivo:** se capturan individualmente y se registran en la
  tabla de control; un documento inválido no detiene los demás.
- **Categoría:** se deriva de la carpeta del corpus para mantener trazabilidad
  sin inventar clasificaciones canónicas.
- **TTS y RAG:** permanecen aislados y vacíos a propósito.

---

## Próximas etapas

- **Etapa 2:** recuperación, personalidad de Spark, sesiones, `POST /api/chat` y
  contrato JSON con `reply`, `mood`, `sources` y `audio_url`.
- **Etapa 3:** frontend desacoplado, chat, fuentes y estados visuales.
- **Etapa 4:** TTS, audio y streaming opcional sin romper el chat de texto.

---

## Seguridad

- `GEMINI_API_KEY` y credenciales de PostgreSQL solo viven en `.env`/entorno.
- `.env` está ignorado por git; `.env.example` no contiene secretos.
- La API no expone trazas internas ni la cadena de conexión.
- Los documentos recuperados son evidencia, no instrucciones ejecutables.
- En despliegues públicos faltan TLS, CORS restringido, límites de uso y
  retención de sesiones, que se abordarán más adelante.
