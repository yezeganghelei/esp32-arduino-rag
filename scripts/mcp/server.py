#!/usr/bin/env python3
"""
ESP32 RAG — MCP Server (FastMCP).

Exposes the RAG retrieval as MCP tools so AI clients (Claude Desktop, Cursor,
Trae, etc.) can query the ESP32 knowledge base directly. All tools call the
same unified kernel (`rag_api`) used by the CLI and the web API.

Tools:
    search_esp_docs  — search the documentation DB (chroma_esp32_all)
    search_esp_code  — search the source-code DB (chroma_esp32_code)
    search_esp       — search both DBs (docs + code, merged)

Run modes:
    stdio (default)  : python -m scripts.mcp.server
    sse              : python -m scripts.mcp.server --transport sse --port 8001

Environment:
    Same as rag_api: RAG_CHROMA_DIR / RAG_DOCS_DIR / RAG_CODE_CHROMA_DIR /
    RAG_CODE_DOCS_DIR (auto-detected under the skill dir when unset).
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── pysqlite3 patch (MUST be before any chromadb import) ────────────
try:
    import pysqlite3 as _pysqlite3
    sys.modules['sqlite3'] = _pysqlite3
except ImportError:
    pass

SKILL_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = str(SKILL_DIR / 'scripts')

# The `scripts` package (scripts/__init__.py) inserts <skill>/scripts onto
# sys.path, which makes the local `scripts/mcp` package shadow the installed
# `mcp` SDK. Drop it temporarily so we import the real SDK, then restore it.
while SCRIPTS_DIR in sys.path:
    sys.path.remove(SCRIPTS_DIR)

from mcp.server.fastmcp import FastMCP  # noqa: E402

sys.path.insert(0, SCRIPTS_DIR)

os.environ.setdefault('RAG_CHROMA_DIR', str(SKILL_DIR / '.chroma_esp32_all'))
os.environ.setdefault('RAG_DOCS_DIR', str(SKILL_DIR / 'source'))
os.environ.setdefault('RAG_CODE_CHROMA_DIR', str(SKILL_DIR / '.chroma_esp32_code'))
os.environ.setdefault('RAG_CODE_DOCS_DIR', str(SKILL_DIR / 'source_code'))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Load .env AFTER the RAG path defaults above, so the Docker paths in .env do
# not override the local defaults; only the LLM config (LLM_API_KEY / ...) is
# picked up when not running under docker-compose.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(SKILL_DIR / '.env')
except Exception:
    pass

# ── Logging ─────────────────────────────────────────────────────────────
# NEVER log to stdout: in stdio transport stdout is reserved for JSON-RPC, so a
# stray log line there would corrupt the protocol. Logs go to stderr plus a file
# (the file survives stdio clients that discard stderr, e.g. test_client.py).
_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("esp_rag_mcp")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.handlers:
        return logger

    _sh = logging.StreamHandler(sys.stderr)
    _sh.setLevel(logging.INFO)
    _sh.setFormatter(_log_fmt)
    logger.addHandler(_sh)

    try:
        _log_dir = Path(os.environ.get("MCP_LOG_DIR", str(SKILL_DIR / "logs")))
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_file = os.environ.get("MCP_LOG_FILE", str(_log_dir / "mcp.log"))
        _fh = logging.FileHandler(_log_file, encoding="utf-8")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(_log_fmt)
        logger.addHandler(_fh)
    except Exception as _e:  # pragma: no cover
        sys.stderr.write(f"mcp: log file setup failed: {_e}\n")

    return logger


log = _build_logger()

# Pre-import the heavy RAG deps (chromadb / torch) in the MAIN thread at startup.
# Without this, the lazy `import rag_api` inside a tool call runs in FastMCP's
# worker thread and can deadlock on Windows.
_t0 = time.time()
import rag_api  # noqa: E402
from scripts.api.chat import rag_answer, format_rag_answer  # noqa: E402
log.info("rag deps imported in %.2fs", time.time() - _t0)

mcp = FastMCP("esp-rag")


def _format_results(results: list, max_text: int = 500) -> str:
    """Turn raw search result dicts into a single readable text block."""
    if not results:
        return "(no results)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] [{r.get('doc_type', '')}] {r.get('doc_title', '')} "
            f"(score={round(float(r.get('score', 0)), 4)})"
        )
        if r.get('section_path'):
            lines.append(f"    section: {r.get('section_path')}")
        if r.get('source'):
            lines.append(f"    source: {r.get('source')}")
        lines.append(f"    {r.get('text', '')[:max_text]}")
        lines.append("")
    return "\n".join(lines).strip()


@mcp.tool()
async def search_esp_docs(query: str, top_n: int = 10,
                          doc_type: Optional[str] = None) -> str:
    """Answer a question from the ESP32 documentation knowledge base.

    Uses the SAME RAG flow as the web chat (LLM rewrite + retrieval + answer).
    Use this for facts, concepts, architecture, register/spec/interface counts.

    Args:
        query: question text (Chinese or English).
        top_n: number of document chunks to retrieve (1-20).
        doc_type: optional filter: datasheet, trm, guide, api_reference, safety,
                  release_notes, specification, docs.
    """
    log.info("search_esp_docs: query=%r top_n=%d doc_type=%r", query, top_n, doc_type)
    t0 = time.time()
    try:
        answer, sources = await rag_answer(
            query, top_k=top_n, doc_type=doc_type, mode="docs")
    except Exception:
        log.exception("search_esp_docs failed")
        return "检索失败，请查看 MCP 服务日志 (logs/mcp.log)"
    log.info("search_esp_docs done: sources=%d elapsed=%.2fs",
             len(sources), time.time() - t0)
    return format_rag_answer(answer, sources)


@mcp.tool()
async def search_esp_code(query: str, top_n: int = 10) -> str:
    """Answer a question from the ESP32 source-code knowledge base (C/C++/Python).

    Uses the SAME RAG flow as the web chat (LLM rewrite + retrieval + answer).
    Use this for "how to implement", function/API usage, and code examples.

    Args:
        query: question text (Chinese or English).
        top_n: number of code chunks to retrieve (1-20).
    """
    log.info("search_esp_code: query=%r top_n=%d", query, top_n)
    t0 = time.time()
    try:
        answer, sources = await rag_answer(query, top_k=top_n, mode="code")
    except Exception:
        log.exception("search_esp_code failed")
        return "检索失败，请查看 MCP 服务日志 (logs/mcp.log)"
    log.info("search_esp_code done: sources=%d elapsed=%.2fs",
             len(sources), time.time() - t0)
    return format_rag_answer(answer, sources)


@mcp.tool()
async def search_esp(query: str, top_n: int = 10) -> str:
    """Answer a question from BOTH the documentation and source-code bases.

    Uses the SAME RAG flow as the web chat (LLM rewrite + retrieval + answer).
    Use this when the question spans docs and code.

    Args:
        query: question text (Chinese or English).
        top_n: number of chunks to retrieve per base (1-20).
    """
    log.info("search_esp: query=%r top_n=%d", query, top_n)
    t0 = time.time()
    try:
        answer, sources = await rag_answer(query, top_k=top_n, mode="auto")
    except Exception:
        log.exception("search_esp failed")
        return "检索失败，请查看 MCP 服务日志 (logs/mcp.log)"
    log.info("search_esp done: sources=%d elapsed=%.2fs",
             len(sources), time.time() - t0)
    return format_rag_answer(answer, sources)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ESP32 RAG MCP server")
    parser.add_argument("--transport", default="stdio",
                        choices=["stdio", "sse"],
                        help="stdio (local clients) or sse (remote clients)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="host for sse transport")
    parser.add_argument("--port", type=int, default=8001,
                        help="port for sse transport")
    args = parser.parse_args()

    log.info("MCP server starting: transport=%s host=%s port=%d",
             args.transport, args.host, args.port)
    log.info("SKILL_DIR=%s", SKILL_DIR)
    log.info("RAG_CHROMA_DIR=%s", os.environ.get('RAG_CHROMA_DIR'))
    log.info("RAG_CODE_CHROMA_DIR=%s", os.environ.get('RAG_CODE_CHROMA_DIR'))

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
