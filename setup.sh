#!/usr/bin/env bash
# ── ESP-RAG data setup (Linux / macOS) ────────────────────────────
# Downloads the models and prebuilt vector databases, then you can run:
#   docker compose up -d
#
# Usage:
#   ./setup.sh                 # models + documentation DB
#   ./setup.sh --db all        # + source-code DB
#   ./setup.sh --db none       # models only
#   ./setup.sh --force         # overwrite existing DBs
#
# Override the image if you built it locally:
#   ESP_RAG_IMAGE=esp-rag:latest ./setup.sh
set -euo pipefail

IMAGE="${ESP_RAG_IMAGE:-ghcr.io/yezeganghelei/esp32-arduino-rag:latest}"

mkdir -p models .chroma_esp32_all .chroma_esp32_code
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example - set LLM_API_KEY before using chat."
fi

docker run --rm -i \
    -v "$PWD/models:/app/models" \
    -v "$PWD/.chroma_esp32_all:/app/.chroma_esp32_all" \
    -v "$PWD/.chroma_esp32_code:/app/.chroma_esp32_code" \
    -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
    --entrypoint python \
    "$IMAGE" -m scripts.setup_data "$@"
