#!/usr/bin/env python3
"""
RAG Query Agent — single-call entry point.

Runs the full RAG answer flow (LLM rewrite + retrieval + generation), the SAME
flow used by the web chat and the MCP server.

Usage:
    python3 agent.py <query> [options]

Options:
    --top N            Number of chunks to retrieve (default: 10)
    --type <doc_type>  Filter by doc type (datasheet, trm, guide, api, safety, ...)
    --raw              JSON output ({"answer": ..., "sources": [...]})
    --code             Query only the source-code database (config.yaml `code:`)
    --docs-only        Query only the documentation database

By default a normal query searches BOTH the documentation database and the
source-code database and returns a grounded answer with source citations.

Environment:
    RAG_CHROMA_DIR  — ChromaDB persist directory
    RAG_DOCS_DIR    — Source document directory
    LLM_API_KEY / LLM_BASE_URL / LLM_MODEL — LLM config (loaded from .env)

Examples:
    python3 agent.py "ESP32-S3 boot process" --top 5
    python3 agent.py "ESP32 和 ESP32-S3 有什么区别？"
    python3 agent.py --raw "CAN bus configuration"
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── pysqlite3 patch (MUST be before any chromadb import) ────────────
try:
    import pysqlite3 as _pysqlite3
    sys.modules['sqlite3'] = _pysqlite3
except ImportError:
    pass

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = SKILL_DIR / 'scripts'
sys.path.insert(0, str(SKILL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault('RAG_CHROMA_DIR', str(SKILL_DIR / '.chroma_esp32_all'))
os.environ.setdefault('RAG_DOCS_DIR', str(SKILL_DIR / 'source'))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Load .env so the LLM config (LLM_API_KEY / LLM_BASE_URL) is available.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(SKILL_DIR / '.env')
except Exception:
    pass

import logging
logging.getLogger('chromadb').setLevel(logging.ERROR)
for name in ['chromadb.segment.impl.metadata', 'chromadb.segment.impl.vector',
             'chromadb.telemetry', 'chromadb.rate_limiting']:
    logging.getLogger(name).setLevel(logging.ERROR)

# ── Logging (stderr + file; stdout is reserved for the query output) ──
_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("esp_rag_agent")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.handlers:
        return logger

    _sh = logging.StreamHandler(sys.stderr)
    _sh.setLevel(logging.INFO)
    _sh.setFormatter(_log_fmt)
    logger.addHandler(_sh)

    try:
        _log_dir = Path(os.environ.get("AGENT_LOG_DIR", str(SKILL_DIR / "logs")))
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_file = os.environ.get("AGENT_LOG_FILE", str(_log_dir / "agent.log"))
        _fh = logging.FileHandler(_log_file, encoding="utf-8")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(_log_fmt)
        logger.addHandler(_fh)
    except Exception as _e:  # pragma: no cover
        sys.stderr.write(f"agent: log file setup failed: {_e}\n")

    return logger


log = _build_logger()

_import_t0 = time.time()
import rag_api
log.info("rag_api imported in %.2fs", time.time() - _import_t0)

from scripts.api.chat import rag_answer, format_rag_answer


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(__doc__.strip())
        sys.exit(0)

    raw = False
    top_n = 10
    doc_type = None
    code_mode = False
    docs_only = False
    query_parts = []

    i = 0
    while i < len(args):
        if args[i] == '--raw':
            raw = True
        elif args[i] == '--code':
            code_mode = True
        elif args[i] == '--docs-only':
            docs_only = True
        elif args[i] == '--top' and i + 1 < len(args):
            top_n = int(args[i + 1])
            i += 1
        elif args[i] == '--type' and i + 1 < len(args):
            doc_type = args[i + 1]
            i += 1
        else:
            query_parts.append(args[i])
        i += 1

    query_text = ' '.join(query_parts).strip()

    if not query_text:
        print("Error: no query provided", file=sys.stderr)
        sys.exit(1)

    mode = 'code' if code_mode else ('docs' if docs_only else 'auto')
    log.info("query: query=%r top_n=%d doc_type=%r mode=%s",
             query_text, top_n, doc_type, mode)
    t0 = time.time()
    import asyncio
    answer, sources = asyncio.run(
        rag_answer(query_text, top_k=top_n, doc_type=doc_type, mode=mode))
    log.info("query done: sources=%d elapsed=%.2fs",
             len(sources), time.time() - t0)
    if raw:
        print(json.dumps({
            "answer": answer,
            "sources": [
                {"doc_title": s.doc_title, "doc_type": s.doc_type,
                 "source": s.source_file, "score": s.score}
                for s in sources
            ],
        }, ensure_ascii=False))
    else:
        print(format_rag_answer(answer, sources))


if __name__ == '__main__':
    main()