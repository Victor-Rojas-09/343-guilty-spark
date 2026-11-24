# Halo RAG — Manual de Halo: Combat Evolved

Sistema RAG (Retrieval-Augmented Generation) en Python para responder preguntas en lenguaje natural
sobre el **manual de Halo: Combat Evolved**. Las respuestas deben:

- Basarse **únicamente** en fragmentos del manual recuperados desde PostgreSQL + `pgvector`.
- **Citar** los fragmentos utilizados (documento y página).
- Indicar **explícitamente** cuando el contexto recuperado no contiene la información.

> **Estado actual: Fase 1 — Scaffolding.** La estructura, la infraestructura de base de datos y los
> contratos de los módulos están definidos. La lógica de ingesta, chunking, embeddings, recuperación
> y generación se implementará en fases posteriores.

---

## Arquitectura de carpetas

```text
.
├── docker-compose.yml          # PostgreSQL + pgvector para desarrollo
├── docker/
│   └── init/
│       └── 01-extensions.sql   # Habilita la extensión vector al iniciar el volumen
├── data/                       # PDF del manual (contenido ignorado por git)
├── eval/                       # Conjunto de evaluación (preguntas y referencias)
│   ├── questions.example.json
│   └── README.md
├── scripts/                    # Puntos de entrada CLI
│   ├── ingest.py               # Ingesta del PDF a la base vectorial (stub)
│   └── ask.py                  # Pregunta en lenguaje natural (stub)
├── src/
│   └── halo_rag/
│       ├── config.py           # Configuración tipada desde variables de entorno
│       ├── ingestion/          # Lectura de PDF, limpieza y chunking
│       ├── db/                 # Conexión, esquema SQL e inicialización de pgvector
│       ├── llm/                # Gemini: embeddings, generación y control de límites
│       ├── retrieval/          # Búsqueda por similitud vectorial
│       ├── rag/                # Orquestación del pipeline y prompts
│       └── eval/               # Arnés de evaluación del sistema
├── tests/                      # Pruebas automatizadas
├── .env.example                # Plantilla de variables de entorno
├── requirements.txt            # Dependencias de ejecución
├── requirements-dev.txt        # Dependencias de desarrollo (tests, lint)
└── pytest.ini                  # Configuración de pytest
```

### Responsabilidad de cada módulo

| Módulo | Responsabilidad |
| --- | --- |
| `ingestion` | Leer el PDF página a página, limpiar el texto y dividirlo en chunks con metadatos (`documento`, `página`, índice). |
| `db` | Abrir conexiones, habilitar `pgvector` y definir/aplicar tablas e índices. |
| `llm` | Cliente del SDK oficial de Google Gemini para embeddings y generación, con lotes y reintentos ante límites de uso. |
| `retrieval` | Ejecutar búsquedas de similitud vectorial en `pgvector` y devolver fragmentos con su fuente. |
| `rag` | Unir recuperación + generación, construir el prompt y garantizar citas y respuesta explícita sin contexto. |
| `eval` | Guardar preguntas directas, multi-fragmento y sin respuesta, y medir la calidad del sistema. |
| `data` | Alojar el PDF del manual. Los PDFs se ignoran en git; usa `data/README` o el `.gitkeep` para conservar la carpeta. |

---

## Requisitos

- Python 3.11+
- Docker y Docker Compose (plugin `docker compose`)
- Una API key gratuita de Google Gemini

---

## Puesta en marcha

### 1. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y completa al menos:

- Credenciales de PostgreSQL (`POSTGRES_*`).
- `GEMINI_API_KEY` con tu clave de Google AI Studio.

> `.env` está ignorado por git. Nunca subas claves al repositorio.

### 2. Levantar PostgreSQL con pgvector

```bash
docker compose up -d
```

Esto crea el contenedor `halo-rag-postgres`, un volumen persistente `pgvector_data` y ejecuta
`docker/init/01-extensions.sql`, que habilita la extensión `vector` en la primera inicialización.

### 3. Verificar que PostgreSQL está operativo

```bash
# Estado del contenedor (debe aparecer "healthy")
docker compose ps

# Logs (opcional)
docker compose logs -f postgres

# Versión del servidor
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT version();"'

# Confirmar que pgvector está instalado (busca "vector" en la salida de \dx)
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dx"'
```

### 4. Preparar el entorno Python (opcional en Fase 1)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # tests y lint
```

### 5. Colocar el manual

Copia el PDF en:

```text
data/halo_combat_evolved_manual.pdf
```

La ruta se puede cambiar con `PDF_PATH` en `.env`. Los PDFs están ignorados por git.

### Detener / reiniciar

```bash
docker compose stop        # detiene sin borrar datos
docker compose down        # elimina contenedores, conserva el volumen
docker compose down -v     # elimina también el volumen (borra la base de datos)
```

---

## Uso previsto (fases siguientes)

```bash
# Ingesta: PDF -> limpieza -> chunks -> embeddings -> pgvector
python scripts/ingest.py

# Pregunta en lenguaje natural con citas
python scripts/ask.py "¿Qué tipos de escudo tiene el Master Chief?"
```

---

## Decisiones técnicas pendientes / por documentar

Estas decisiones se justificarán durante la implementación:

1. **Tamaño de chunk y solapamiento**: valores iniciales `CHUNK_SIZE=800` y `CHUNK_OVERLAP=150`,
   a validar con el conjunto de evaluación. Considerar chunking por párrafos/secciones del manual.
2. **Estrategia de chunking**: por caracteres, tokens o estructura del documento; manejo de tablas,
   listas e imágenes con texto.
3. **Modelo y dimensión de embeddings**: `gemini-embedding-001` con dimensión de salida configurable
   (`EMBEDDING_DIM`, por defecto 768). La dimensión debe coincidir con la columna `vector(N)`.
4. **Métrica de distancia**: coseno (habitual en texto), L2 o producto interno; debe ser coherente con
   la normalización de los embeddings y con el operador del índice.
5. **Índice de `pgvector`**: HNSW vs IVFFlat, parámetros (`m`, `ef_construction`, `lists`, `probes`)
   y umbral de filas a partir del cual conviene crearlo.
6. **Recuperación**: `top_k`, umbral mínimo de similitud, filtros por página/sección y re-ranking.
7. **Extracción de PDF**: `pypdf` como base; evaluar `pdfplumber`/`PyMuPDF` si el texto o las tablas
   no se extraen con calidad.
8. **Control de límites de Gemini**: tamaño de lote, backoff exponencial, concurrencia y presupuesto
   de la capa gratuita.
9. **Respuesta sin contexto**: redacción exacta de la negativa y comportamiento cuando la similitud
   es baja.
10. **Formato de citas**: por ejemplo `[Manual HCE, p. 12]`, incluyendo documento y página.
11. **Evaluación**: métricas (faithfulness, relevancia, precisión/recall del contexto, tasa de
    abstención correcta) y umbrales de aceptación.
12. **Persistencia de documentos**: control de versiones del PDF, checksum e idempotencia de la ingesta.
