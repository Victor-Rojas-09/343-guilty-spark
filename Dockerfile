FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 spark \
    && chown -R spark:spark /app

USER spark

EXPOSE 8000

CMD ["uvicorn", "spark_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
