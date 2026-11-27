# 343 Guilty Spark // Asistente conversacional con RAG

Backend del asistente basado en el corpus documental de **343 Guilty Spark**.
Esta entrega implementa la **Etapa 1 (infraestructura e ingesta documental)**
y la **Etapa 2 (motor RAG, personalidad, sesiones y API)**.

Etapa 1:

- PostgreSQL 16 + `pgvector` reproducible con Docker Compose.
- Configuración centralizada y secretos fuera del repositorio.
- Ingesta recursiva de PDF, Markdown y TXT con LangChain.
- Limpieza, chunking configurable y metadatos de procedencia.
- Embeddings de Gemini (`gemini-embedding-001`) validados antes de persistir.
- Persistencia vectorial con `langchain-postgres`.
- Control SHA256 por archivo para evitar duplicados y reindexar cambios.
- `GET /health`.

Etapa 2:

- Recuperación semántica en pgvector con top-k, umbral de similitud y filtros.
- Jerarquía de fuentes (canon, transcripciones, referencia, comunidad).
- System Prompt versionado de 343 Guilty Spark y tratamiento de lagunas.
- Cadena RAG de LangChain con salida estructurada `{reply, mood}`.
- Sesiones e historial persistidos en PostgreSQL, aislados por `session_id`.
- `POST /api/chat` con contrato JSON `reply`, `mood`, `sources`, `audio_url`.
- Set de preguntas de evaluación del personaje y registro de resultados.

El frontend (Etapa 3) y el TTS (Etapa 4) quedan para las etapas siguientes;
`src/spark_rag/tts/` se mantiene como placeholder aislado y `audio_url` es
siempre `null`.

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
│   ├── test_search.py
│   └── evaluate.py
├── reports/
│   └── eval/                 # Informes de evaluación del personaje
├── src/
│   └── spark_rag/
│       ├── __init__.py
│       ├── config.py
│       ├── db/
│       │   ├── __init__.py
│       │   ├── connection.py
│       │   ├── models.py
│       │   └── sessions.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   ├── cleaner.py
│       │   ├── metadata.py
│       │   ├── embeddings.py
│       │   └── pipeline.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── chat.py
│       │   └── schemas.py
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── retriever.py
│       │   ├── sources.py
│       │   ├── prompts.py
│       │   ├── llm.py
│       │   ├── llm_errors.py
│       │   ├── structured.py
│       │   ├── fallback.py
│       │   ├── chain.py
│       │   ├── mood.py
│       │   ├── history.py
│       │   ├── service.py
│       │   ├── errors.py
│       │   └── evaluation.py
│       └── tts/
│           └── __init__.py
└── tests/
    ├── unit/
    ├── integration/
    └── eval/
        └── spark_questions.json
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
| `db/sessions.py` | Tablas `chat_sessions`/`chat_messages` y repositorio de sesiones. |
| `rag/retriever.py` | Búsqueda en pgvector: top-k, umbral de similitud y filtros por categoría/fuente. |
| `rag/sources.py` | Clase de autoridad de cada fragmento, `sources`/`source_details` y contexto delimitado. |
| `rag/prompts.py` | System Prompt versionado (`PROMPT_VERSION`) y construcción de mensajes. |
| `rag/llm.py` | Fábrica del modelo de chat según `LLM_PROVIDER` (`gemini` u `openai_compat`). |
| `rag/llm_errors.py` | Clasificación de errores del modelo: cuáles activan el respaldo. |
| `rag/structured.py` | Esquema `{reply, mood}`, modos de salida estructurada y limpieza del razonamiento. |
| `rag/fallback.py` | Cadena principal + respaldos (`LLM_FALLBACK_PROVIDERS`) con registro de intentos. |
| `rag/chain.py` | Cadena LangChain: recuperación → prompt → salida estructurada `{reply, mood}`. |
| `rag/mood.py` | Conjunto controlado de `mood` y normalización. |
| `rag/history.py` | Turnos de conversación y recorte del historial enviado al modelo. |
| `rag/service.py` | Orquestación de un turno: sesión, historial, cadena y persistencia. |
| `rag/errors.py` | Errores de dominio que la API traduce a respuestas controladas. |
| `rag/evaluation.py` | Comprobaciones del set de evaluación del personaje. |
| `api/main.py` | Aplicación FastAPI, `GET /health` y manejo de errores sin trazas. |
| `api/chat.py` | `POST /api/chat`. |
| `api/schemas.py` | Contrato JSON de solicitud, respuesta y error. |

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
| `LLM_PROVIDER` | Proveedor principal del chat (`gemini` u `openai_compat`). |
| `LLM_FALLBACK_PROVIDERS` | Respaldo del chat en orden (CSV); por defecto `openai_compat`. Vacío = sin respaldo. |
| `OPENAI_COMPAT_*` | Proveedor compatible con OpenAI usado como respaldo (NVIDIA). |
| `GEMINI_CHAT_MODEL` | Modelo de generación de Gemini. |
| `LLM_TEMPERATURE` / `LLM_MAX_OUTPUT_TOKENS` | Creatividad y longitud máxima de la respuesta. |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | Tiempo máximo y reintentos por llamada al modelo. |
| `RETRIEVAL_TOP_K` | Fragmentos recuperados por pregunta. |
| `RETRIEVAL_MIN_SIMILARITY` | Similitud coseno mínima (1 − distancia) para usar un fragmento. |
| `RETRIEVAL_CATEGORIES` | CSV de categorías permitidas; vacío = todas. |
| `CHAT_HISTORY_MAX_MESSAGES` / `CHAT_HISTORY_MAX_CHARS` | Límite del historial enviado al modelo. |
| `CHAT_MESSAGE_MAX_CHARS` | Longitud máxima del mensaje del usuario. |

Todas las variables se leen en `config.py` y también se pasan al servicio
`api` desde `docker-compose.yml`.

`.env` está ignorado por git. El repositorio solo incluye valores de ejemplo no
secretos en `.env.example`.

### Respaldo del chat con NVIDIA (activo por defecto)

El chat usa Gemini como proveedor principal y, si Gemini falla por cuota (429),
saturación (5xx), timeout o conexión, responde el modelo de NVIDIA
(`mistralai/mistral-large-4-1t-a52b`, API compatible con OpenAI). Los
embeddings siguen siendo siempre de Gemini. `.env.example` trae el respaldo
activado (`LLM_FALLBACK_PROVIDERS=openai_compat`); solo falta la clave:

1. Entra en <https://build.nvidia.com>, inicia sesión y crea una clave en
   **API Keys** (empieza por `nvapi-`).
2. Pégala en tu `.env` (nunca en `.env.example`):

   ```bash
   OPENAI_COMPAT_API_KEY=nvapi-...
   ```

3. Reinicia la API: `docker compose up -d api`.

Si el respaldo está activo y falta la clave, la API arranca igualmente
(`/health` responde), pero registra al arrancar un error que dice qué hacer:

```text
Chat sin configurar: El respaldo del chat 'openai_compat' (LLM_FALLBACK_PROVIDERS) no está configurado. OPENAI_COMPAT_API_KEY no está configurada. Crea una clave en https://build.nvidia.com (API Keys) y añádela a .env como OPENAI_COMPAT_API_KEY=<tu clave>; después reinicia la API. Si no quieres usar el respaldo, deja LLM_FALLBACK_PROVIDERS vacío en .env.
```

y `POST /api/chat` responde `503 llm_not_configured` hasta corregirlo.

Para desactivar el respaldo, deja la variable vacía en `.env`
(`LLM_FALLBACK_PROVIDERS=`) y reinicia la API: el chat usará solo Gemini, como
antes. Si tu `.env` no tiene la variable, Docker Compose también la pasa vacía
(sin respaldo). Con el respaldo activo, los mensajes y los fragmentos
recuperados del corpus se envían también a NVIDIA cuando Gemini falla. Detalles
de timeouts y cooldown en la sección «Respaldo del modelo de chat» de las
decisiones de la Etapa 2 y en los comentarios de `.env.example`.

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
  "version": "0.2.0",
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

## Chat: `POST /api/chat` (Etapa 2)

### Contrato

Solicitud:

```json
{
  "message": "¿Qué relación existe entre Chakas y tú?",
  "session_id": "3f0c9a52-6a1e-4d1b-9a7e-2a4f3c1b5d10"
}
```

- `message`: obligatorio, no vacío, máximo `CHAT_MESSAGE_MAX_CHARS`.
- `session_id`: opcional. Si se omite, el backend crea una sesión nueva y
  devuelve su UUID; las siguientes llamadas deben reenviarlo.

Respuesta `200`:

```json
{
  "reply": "...respuesta en personaje...",
  "mood": "thinking",
  "sources": ["history/01_Dossier_de_lore_343_Guilty_Spark.pdf"],
  "audio_url": null,
  "session_id": "3f0c9a52-6a1e-4d1b-9a7e-2a4f3c1b5d10",
  "source_details": [
    {
      "source": "history/01_Dossier_de_lore_343_Guilty_Spark.pdf",
      "category": "history",
      "page": 2,
      "source_class": "canon",
      "similarity": 0.8521,
      "work": null, "edition": null, "level": null,
      "terminal": null, "scene": null, "speaker": null
    }
  ]
}
```

- `mood` pertenece siempre al conjunto `idle | thinking | enthusiastic |
  rampant`. Lo propone el modelo mediante salida estructurada y el backend lo
  valida; un valor inválido se convierte en `thinking`.
- `sources` mantiene el formato del manual (nombres de archivo). Si no hay
  evidencia suficiente, `sources` y `source_details` van vacíos: nunca se
  inventan referencias.
- `source_details` añade página, categoría, clase de autoridad, similitud y,
  solo si existen en un sidecar, obra/edición/nivel/terminal/escena/hablante.
- `audio_url` es `null` hasta la Etapa 4.

### Errores

Todos los errores usan el cuerpo `{"error": "...", "detail": "..."}` sin
trazas, rutas ni credenciales.

| Código | `error` | Cuándo |
| --- | --- | --- |
| 422 | `invalid_request` | JSON inválido, `message` vacío/ausente/largo, `session_id` que no es UUID. |
| 404 | `session_not_found` | `session_id` con formato válido que no existe. |
| 503 | `llm_not_configured` | Falta `GEMINI_API_KEY` o el proveedor no está soportado. |
| 503 | `llm_unavailable` | Gemini falla, agota la cuota, no responde a tiempo o devuelve vacío. |
| 503 | `retrieval_unavailable` | pgvector o los embeddings de la consulta no están disponibles. |
| 503 | `storage_unavailable` | No se puede leer o guardar el historial. |
| 500 | `internal_error` | Error no previsto (el detalle solo va al log del backend). |

### Ejemplo

PowerShell:

```powershell
$body = @{ message = "¿Qué relación existe entre Chakas y tú?" } | ConvertTo-Json
$r = Invoke-RestMethod -Uri http://localhost:8000/api/chat -Method Post `
  -ContentType "application/json; charset=utf-8" -Body $body
$r.reply
# Segundo turno en la misma sesión
$body = @{ message = "¿Y qué recuerdas de Erde-Tyrene?"; session_id = $r.session_id } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/chat -Method Post `
  -ContentType "application/json; charset=utf-8" -Body $body
```

bash:

```bash
curl -s http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué relación existe entre Chakas y tú?"}'
```

### Cómo responde Spark

1. **Sesión:** se crea o se valida el `session_id` y se cargan sus últimos
   `CHAT_HISTORY_MAX_MESSAGES` mensajes (máximo `CHAT_HISTORY_MAX_CHARS`).
2. **Recuperación:** se busca en pgvector con el mensaje actual más el último
   mensaje del usuario (para que las preguntas de seguimiento encuentren
   contexto), con `RETRIEVAL_TOP_K`, `RETRIEVAL_MIN_SIMILARITY` y
   `RETRIEVAL_CATEGORIES`.
3. **Contexto:** los fragmentos se ordenan por autoridad y se envuelven en
   bloques `<documento>` con su procedencia; el texto se escapa para que no
   pueda inyectar instrucciones.
4. **Generación:** System Prompt de Spark + historial + evidencia → Gemini con
   salida estructurada `{reply, mood}`.
5. **Persistencia:** el mensaje del usuario y la respuesta se guardan juntos en
   una transacción solo si la generación tuvo éxito, con `mood`, fuentes,
   versión del prompt y modelo.

### Jerarquía de fuentes

La clase de autoridad se deriva de la carpeta del corpus (o de
`canon_status`, si un sidecar lo define):

| Carpeta | `source_class` | Tratamiento en el prompt |
| --- | --- | --- |
| `history/` | `canon` | Dossier basado en canon oficial; base para afirmar hechos. |
| `dialogs/` | `transcripciones` | Localizador de escenas; citar obra/nivel y no reproducir guiones. |
| `sources/` | `referencia` | Catálogo y referencias secundarias (Halopedia). |
| `community/` | `comunidad` | Interpretación de fans: nunca se presenta como canon. |

### Personaje

`rag/prompts.py` fija la etapa temporal por defecto: **Spark actual** (tras su
arco con Rion Forge), que recuerda toda su cronología —Chakas, Instalación 07,
conversión, aislamiento, Halo CE, Threshold/Oráculo, Halo 3 y fragmentos
posteriores— con la voz clásica del Monitor. Chakas, 343 Guilty Spark, el
Oráculo y Spark se tratan como etapas de una misma existencia con matices.
Desde `spark-v3` habla desde dentro de su mundo: traduce los títulos de juegos
y novelas del corpus y el nombre "Jefe Maestro" a sus términos (el Reclamador,
la Instalación 04, el Arca), salvo cuando se le piden fuentes o localizadores
o sirve para ubicar de dónde sale un hecho (`spark-v5`),
y se ciñe a los fragmentos sin fusionar datos ni añadir fechas o causas.
Cada cambio del prompt debe incrementar `PROMPT_VERSION`.

---

## Evaluación del personaje

`tests/eval/spark_questions.json` contiene las preguntas del manual (§4.3 y
T2-01…T2-05): identidad, protocolo, aislamiento, Halo 3, evolución, fuentes,
laguna, diálogo, un hecho cubierto, un localizador, una pregunta de seguimiento
y una fuera de dominio.

```bash
# Con el corpus ingerido, PostgreSQL accesible y GEMINI_API_KEY configurada
python scripts/evaluate.py
```

Entre pregunta y pregunta espera 15 s por defecto (`--pause`, mínimo 15) para
no superar el límite de 5 peticiones por minuto del plan gratuito de Gemini.

El script usa la cadena RAG real con la cadena de chat configurada
(`LLM_PROVIDER` y, si hay, `LLM_FALLBACK_PROVIDERS`), aplica comprobaciones
automáticas (fuentes esperadas, marcadores de laguna, menciones, longitud
máxima) y guarda un informe en `reports/eval/eval_<fecha>_<prompt>.json` con:
fecha, versión de código, commit, versión del prompt, cadena de chat
(`chat_chain`, `chat_generation` con temperatura, `top_p`, modo estructurado y
timeout de cada proveedor, cooldown del principal), modelo de embeddings,
documentos indexados, parámetros de recuperación y PASS/FAIL por pregunta y
por modelo (`summary_by_model`).

Cada pregunta registra el modelo que respondió (`model`), si se usó el
respaldo (`fallback_used`), el modo y si la salida fue estructurada, el mood
tal como lo propuso el modelo (`mood_raw`) y si hubo que corregirlo
(`mood_corrected`), y los intentos (`attempts`: modelo, resultado, segundos y,
cuando el modelo los informa, tokens de entrada, salida y razonamiento). Cada resultado incluye
`manual_review: "pendiente"`: las comprobaciones automáticas son orientativas y
el comportamiento esperado se revisa a mano. En `compliance`, el criterio
«Gemini genera respuestas con la personalidad acordada» queda `PENDIENTE`
mientras ninguna respuesta sea de Gemini (las de otro proveedor validan el
pipeline, no ese criterio) y pasa a `REVISION_MANUAL` cuando las hay; nunca se
da por cumplido automáticamente.

Repite la evaluación después de cambiar documentos, prompt, modelo o
parámetros de recuperación (pruebas de regresión del manual §4.2).

---

## Pruebas

```bash
# Pruebas unitarias (no requieren base de datos ni Gemini)
pytest tests/unit

# Pruebas de integración con PostgreSQL/pgvector real
RUN_INTEGRATION=1 pytest tests/integration
```

Las pruebas de integración se omiten si `RUN_INTEGRATION` no vale `1`. No se
hacen llamadas reales a Gemini en las pruebas unitarias ni de integración
(la recuperación de integración usa embeddings falsos deterministas en una
colección temporal).

En PowerShell:

```powershell
pytest tests/unit
$env:RUN_INTEGRATION=1; pytest tests/integration
ruff check .
```

Cobertura de la Etapa 2:

- Configuración nueva y sus validaciones.
- Contrato JSON (`reply`, `mood`, `sources`, `audio_url`) y `mood` controlado.
- Jerarquía de fuentes, deduplicación y contexto delimitado anti-inyección.
- Recuperador: umbral, top-k y filtros (unitarias y contra pgvector real).
- Contenido del System Prompt y separación sistema/historial/evidencia.
- Fábrica del modelo de chat y errores de configuración.
- Cadena RAG con dobles: con/sin evidencia, salida no estructurada y fallos.
- Sesiones: persistencia, orden, límite e aislamiento (PostgreSQL real).
- Servicio y `POST /api/chat`: sesiones nuevas/existentes/inexistentes,
  validación, fallos de Gemini/BD y ausencia de trazas o secretos.
- Comprobaciones del set de evaluación.

Cobertura de la Etapa 1:

- Limpieza y normalización de texto.
- Construcción de metadatos y sidecars.
- Hash SHA256 e IDs estables de chunks.
- Descubrimiento recursivo de formatos.
- Validación de vectores (dimensión, vacío y valores no finitos).
- Configuración y URLs de conexión.
- Disponibilidad de PostgreSQL/pgvector y esquema idempotente.

---

## Decisiones de la Etapa 1

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
- **TTS:** permanece aislado y vacío a propósito.

## Decisiones de la Etapa 2

- **Etapa temporal del personaje:** Spark actual con memoria de toda su
  cronología (manual §2.5); la recuperación aporta el contexto histórico.
- **Modelo de chat:** `gemini-3.8-flash`. `gemini-2.5-flash` devuelve 404 para
  claves nuevas; `gemini-3.8-flash` es el modelo que recomienda la API, tiene
  versión fija (a diferencia de los alias `*-latest`, que cambian sin aviso y
  romperían la comparabilidad de las evaluaciones) y respeta la salida
  estructurada. Su latencia es muy variable (medidas de 3 a 65 s), por eso
  `LLM_TIMEOUT_SECONDS=90`. Se sobrescribe con `GEMINI_CHAT_MODEL` en `.env`.
- **Umbral de similitud:** `RETRIEVAL_MIN_SIMILARITY=0.68`. Con
  `gemini-embedding-001` las similitudes están comprimidas: en el corpus
  actual los fragmentos pertinentes puntúan ~0.69-0.86 y las consultas ajenas
  al dominio ~0.64-0.67. El umbral filtra lo ajeno; las lagunas dentro del
  universo Halo las gestiona el prompt declarando el límite en personaje.
- **`mood`:** salida estructurada del modelo con enum y valor por defecto
  `thinking` si no se respeta el esquema.
- **`sources`:** lista de nombres según el manual + `source_details` con
  metadatos para la Etapa 3.
- **Sesiones:** tablas propias (`chat_sessions`, `chat_messages`) en lugar de
  `PostgresChatMessageHistory` para guardar última actividad, `mood`, fuentes,
  versión de prompt y modelo por mensaje (§5.1). Un `session_id` desconocido
  devuelve 404; no se crea en silencio.
- **Consulta de recuperación:** mensaje actual + último mensaje del usuario,
  sin una llamada extra al modelo para reescribir la pregunta.
- **Retención de sesiones:** sin política de borrado todavía; debe decidirse
  antes de abrir el uso a terceros.

---

## Próximas etapas

- **Etapa 3:** frontend desacoplado, chat, fuentes y estados visuales.
- **Etapa 4:** TTS, audio y streaming opcional sin romper el chat de texto.

---

## Seguridad

- `GEMINI_API_KEY` y credenciales de PostgreSQL solo viven en `.env`/entorno.
- `.env` está ignorado por git; `.env.example` no contiene secretos.
- La API no expone trazas internas ni la cadena de conexión.
- Los documentos recuperados son evidencia, no instrucciones ejecutables: se
  envían al modelo en bloques `<documento>` escapados, separados del System
  Prompt.
- Los errores de `/api/chat` no incluyen trazas, cadenas de conexión ni claves;
  el servicio de chat no escribe en los logs el contenido de los mensajes.
- En despliegues públicos faltan TLS, CORS restringido, límites de uso y
  retención de sesiones, que se abordarán más adelante.
