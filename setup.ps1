# ── ESP-RAG data setup (Windows / PowerShell) ─────────────────────
# Downloads the models and prebuilt vector databases, then you can run:
#   docker compose up -d
#
# Usage:
#   .\setup.ps1                  # models + documentation DB
#   .\setup.ps1 --db all         # + source-code DB
#   .\setup.ps1 --db none        # models only
#   .\setup.ps1 --corpus         # also fetch source/ + source_code/ corpus
#   .\setup.ps1 --force          # overwrite existing DBs / corpus
#
# Override the image if you built it locally:
#   $env:ESP_RAG_IMAGE = "esp-rag:latest"; .\setup.ps1
$ErrorActionPreference = "Stop"

$image = $env:ESP_RAG_IMAGE
if (-not $image) { $image = "ghcr.io/yezeganghelei/esp32-arduino-rag:latest" }

New-Item -ItemType Directory -Force -Path models, .chroma_esp32_all, .chroma_esp32_code, source, source_code | Out-Null
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example - set LLM_API_KEY before using chat."
}

docker run --rm -i `
    -v "${PWD}/models:/app/models" `
    -v "${PWD}/.chroma_esp32_all:/app/.chroma_esp32_all" `
    -v "${PWD}/.chroma_esp32_code:/app/.chroma_esp32_code" `
    -v "${PWD}/source:/app/source" `
    -v "${PWD}/source_code:/app/source_code" `
    -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 `
    --entrypoint python `
    $image -m scripts.setup_data @args
