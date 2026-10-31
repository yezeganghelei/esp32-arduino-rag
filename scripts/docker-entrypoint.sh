#!/bin/bash
# ── Docker entrypoint ──────────────────────────────────────────────
# Modes:
#   server           → start FastAPI server (port 8000)
#   web              → start Gradio web UI (port 7860)
#   all              → start both API server + web UI
#   build            → run incremental index build
#   anything else    → pass through to agent.py (query mode)

set -e

cd /app

case "$1" in
    server)
        shift
        exec uvicorn scripts.api.server:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers 1 \
            "$@"
        ;;
    web)
        shift
        exec python -m scripts.web.app "$@"
        ;;
    all)
        shift
        # Start API server in background
        uvicorn scripts.api.server:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers 1 &
        API_PID=$!

        # Start web UI in foreground
        python -m scripts.web.app "$@" &
        WEB_PID=$!

        # Wait for any process to exit
        wait -n $API_PID $WEB_PID
        EXIT_CODE=$?

        # Clean up the other process
        kill $API_PID $WEB_PID 2>/dev/null || true
        wait $API_PID $WEB_PID 2>/dev/null || true

        exit $EXIT_CODE
        ;;
    build)
        shift
        exec python -m scripts.build.run "$@"
        ;;
    *)
        exec python scripts/agent.py "$@"
        ;;
esac
