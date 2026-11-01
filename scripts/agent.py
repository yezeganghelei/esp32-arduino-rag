#!/usr/bin/env python3
"""
RAG Query Agent — single-call entry point.

Performs pure search and returns raw results. No intent analysis — that is
handled by the AI (the LLM you're talking to).

Usage:
    python3 agent.py <query> [options]

Options:
    --top N            Return top N results (default: 10)
    --type <doc_type>  Filter by doc type (datasheet, trm, guide, api, safety, ...)
    --raw              JSON lines output (machine-readable)
    --code             Query only the source-code database (config.yaml `code:`)
    --docs-only        Query only the documentation database

By default a normal query searches BOTH the documentation database and the
source-code database (when it exists) and merges the results. All queries go
through raw retrieval (no spec/comparison detection).

Environment:
    RAG_CHROMA_DIR  — ChromaDB persist directory
    RAG_DOCS_DIR    — Source document directory

Examples:
    python3 agent.py "boot process" --top 5
    python3 agent.py "chip errata" --type docs
    python3 agent.py --raw "CAN bus configuration"
"""

import json
import os
import re
import sys
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

import logging
logging.getLogger('chromadb').setLevel(logging.ERROR)
for name in ['chromadb.segment.impl.metadata', 'chromadb.segment.impl.vector',
             'chromadb.telemetry', 'chromadb.rate_limiting']:
    logging.getLogger(name).setLevel(logging.ERROR)

import rag_api

# Source-file reading helpers now live in rag_api (shared with the web API).
read_source_file = rag_api.read_source_file
resolve_source_path = rag_api.resolve_source_path

# ═══════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════

def format_answer(results, query_text: str = "", raw: bool = False) -> str:
    if not results:
        msg = "No relevant results found."
        return json.dumps({"query": query_text, "results": []}) if raw else msg

    if results[0].get('type') == 'comparison':
        table = results[0]['table']
        if raw:
            return json.dumps({"type": "comparison", "table": table})
        return table

    if raw:
        output = []
        for r in results:
            output.append(json.dumps({
                'score': round(r.get('score', 0), 4),
                'doc_type': r.get('doc_type', ''),
                'source': r.get('source', ''),
                'section': r.get('section_path', ''),
                'doc_title': r.get('doc_title', ''),
                'text': r.get('text', '')[:800],
            }, ensure_ascii=False))
        return '\n'.join(output)

    lines = []
    if query_text:
        lines.append(f"## Results for: _{query_text}_\n")
    lines.append(f"Found **{len(results)}** relevant passages:\n")

    for i, r in enumerate(results, 1):
        score = r.get('score', 0)
        dtype = r.get('doc_type', '?')
        source = r.get('source', '?')
        section = r.get('section_path', '')
        title = r.get('doc_title', '')
        text = r.get('text', '')

        lines.append(f"### {i}. [{dtype}] {title or Path(source).stem}")
        lines.append(f"**Source**: `{source}`")
        if section:
            lines.append(f"**Section**: {section}")
        lines.append(f"**Relevance**: {score:.3f}\n")
        lines.append(text[:600].strip())
        lines.append("")

        src_content = read_source_file(source, extra_root=r.get('_src_root'))
        if src_content:
            lines.append("> _Excerpt from source file:_")
            excerpt = re.sub(r'\n{3,}', '\n\n', src_content[:1500]).strip()
            lines.append(f"> {excerpt[:800]}")
            lines.append("")

    return '\n'.join(lines)


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
    results, _ = rag_api.search(query_text, top_n=top_n, doc_type=doc_type,
                                mode=mode)
    print(format_answer(results, query_text, raw=raw))


if __name__ == '__main__':
    main()