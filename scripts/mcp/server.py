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

import os
import sys
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
def search_esp_docs(query: str, top_n: int = 10,
                    doc_type: Optional[str] = None) -> str:
    """Search the ESP32 documentation knowledge base (datasheets, TRM, guides).

    Use this for facts, concepts, architecture, register/spec/interface counts.

    Args:
        query: search text (Chinese or English). Use English technical terms
               (e.g. "clock source PLL XTAL") for best recall.
        top_n: number of results to return (1-20).
        doc_type: optional filter: datasheet, trm, guide, api_reference, safety,
                  release_notes, specification, docs.
    """
    import rag_api
    results, _ = rag_api.search(query, top_n, doc_type, 'docs')
    return _format_results(results)


@mcp.tool()
def search_esp_code(query: str, top_n: int = 10) -> str:
    """Search the ESP32 source-code knowledge base (C/C++/Python examples).

    Use this for "how to implement", function/API usage, and code examples.
    Prefer code identifiers in the query (e.g. "esp_camera_init camera_config_t").

    Args:
        query: search text (Chinese or English).
        top_n: number of results to return (1-20).
    """
    import rag_api
    results, _ = rag_api.search(query, top_n, None, 'code')
    return _format_results(results)


@mcp.tool()
def search_esp(query: str, top_n: int = 10) -> str:
    """Search BOTH the documentation and source-code knowledge bases (merged).

    Use this when the question spans docs and code (e.g. "how to configure X"
    which needs both the explanation and the code).

    Args:
        query: search text (Chinese or English).
        top_n: number of results to return (1-20).
    """
    import rag_api
    results, _ = rag_api.search(query, top_n, None, 'auto')
    return _format_results(results)


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

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
