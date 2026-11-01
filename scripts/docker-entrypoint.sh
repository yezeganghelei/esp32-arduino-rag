#!/bin/bash
# ── Docker entrypoint ──────────────────────────────────────────────
# Modes:
#   server           → start FastAPI server (port 8000)
#   web              → start Gradio web UI (port 7860)
#   mcp              → start MCP server (SSE, port 8001)
#   all              → start API server + web UI + MCP server
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
    mcp)
        shift
        exec python -m scripts.mcp.server --transport sse --host 0.0.0.0 --port 8001 "$@"
        ;;
    all)
        shift
        # Start API server in background
        uvicorn scripts.api.server:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers 1 &
        API_PID=$!

        # Start MCP server in background
        python -m scripts.mcp.server --transport sse --host 0.0.0.0 --port 8001 &
        MCP_PID=$!

        # Start web UI in foreground
        python -m scripts.web.app "$@" &
        WEB_PID=$!

        # Wait for any process to exit
        wait -n $API_PID $MCP_PID $WEB_PID
        EXIT_CODE=$?

        # Clean up the other processes
        kill $API_PID $MCP_PID $WEB_PID 2>/dev/null || true
        wait $API_PID $MCP_PID $WEB_PID 2>/dev/null || true

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
