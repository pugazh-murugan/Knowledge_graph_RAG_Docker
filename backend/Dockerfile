FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code
COPY . .

# Run as a non-root user. data/ and graph_data/ are meant to be mounted as volumes.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/graph_data \
    && chown -R appuser:appuser /app
USER appuser

ENV PDF_PATH=/app/data/spotify_web_app_architecture.pdf \
    GRAPH_PATH=/app/graph_data/kg.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
