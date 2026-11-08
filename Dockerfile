# ── ESP32 RAG Dockerfile (Scheme A: data mounted externally) ──────
#
# This image contains ONLY code and dependencies.
# Models, vector DB, and source documents are mounted as volumes.
#
# Build:
#   docker build -t esp-rag:latest .
#
# Run (basic query):
#   docker run --rm \
#     -v /path/to/models:/app/models \
#     -v /path/to/source:/app/source \
#     -v /path/to/.chroma_esp32_all:/app/.chroma_esp32_all \
#     esp-rag:latest "ESP32-S3 boot process" --top 5
#
# Run (FastAPI server):
#   docker run --rm -p 8000:8000 \
#     -v /path/to/models:/app/models \
#     -v /path/to/source:/app/source \
#     -v /path/to/.chroma_esp32_all:/app/.chroma_esp32_all \
#     esp-rag:latest server

FROM python:3.10-slim AS base

# ── System dependencies ────────────────────────────────────────────
# PyMuPDF and chromadb need some system libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── pysqlite3 ──────────────────────────────────────────────────────
# ChromaDB on Linux needs a newer sqlite3 than the one bundled with
# Python. pysqlite3-binary provides a pre-built wheel.
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip pysqlite3-binary

# ── Python dependencies ────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# ── App code ───────────────────────────────────────────────────────
COPY scripts/ ./scripts/
COPY config.yaml .

# ── Runtime environment ────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    RAG_CHROMA_DIR=/app/.chroma_esp32_all \
    RAG_DOCS_DIR=/app/source \
    MODELS_DIR=/app/models

# pysqlite3 shim is applied at runtime by main.py / agent.py.

EXPOSE 8000 7860

# ── Entrypoint ─────────────────────────────────────────────────────
# Modes:
#   - "server"   : start FastAPI server (port 8000)
#   - "web"      : start Gradio web UI (port 7860)
#   - "all"      : start both API server and web UI
#   - anything   : pass through to agent.py (query mode)
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["server"]
