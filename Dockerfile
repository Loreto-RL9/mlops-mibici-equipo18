# Imagen del servicio de pronóstico de MiBici (FastAPI + Pipeline @champion).
# Funciona igual con Docker y con Podman (antes hay que exportar el campeón: make export-model):
#   docker build -t mibici-api:latest .
#   docker run -p 8000:8000 mibici-api:latest        # http://127.0.0.1:8000/docs
# Publicación en Docker Hub con tags versionados: make publish DOCKERHUB_USER=<usuario>
#
# Dos etapas: la primera instala las dependencias exactas de uv.lock con uv; la segunda copia sólo
# el entorno, el código, el catálogo de estaciones y el modelo, y corre como usuario sin root.

ARG PYTHON_VERSION=3.12

# --- 1. Construcción: dependencias fijadas en uv.lock (sin las de desarrollo) ---------------------
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir "uv==0.11.32"
WORKDIR /app
# Sólo las dependencias: la capa se reutiliza mientras no cambie uv.lock. El paquete mibici no se
# instala: se copia tal cual en la etapa 2 y config.py ubica datos y modelos relativos a /app.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --- 2. Ejecución: imagen mínima, usuario sin root ------------------------------------------------
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="mibici-api" \
      org.opencontainers.image.description="Pronóstico de salidas por estación y hora (MiBici, Equipo 18)" \
      org.opencontainers.image.source="https://github.com/Loreto-RL9/mlops-mibici-equipo18"
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MIBICI_MODEL_DIR=/app/models/champion \
    MIBICI_PREDICTION_LOG=/app/logs/predicciones.jsonl
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin mibici
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY mibici ./mibici
COPY data/external/nomenclatura_2026_09.csv ./data/external/
COPY models/champion ./models/champion
# config.py crea las carpetas de datos, modelos y reportes al importarse; el usuario debe poder
# escribirlas (y el log de eventos), pero no el código ni el modelo
RUN mkdir -p data/raw data/interim data/processed reports/figures logs \
    && chown -R mibici:mibici data reports logs
USER mibici
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "mibici.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
