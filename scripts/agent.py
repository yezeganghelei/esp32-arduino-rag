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
    --spec <name>      Look up structured spec data for named SoC
    --compare <A,B>    Compare two SoCs by name

Environment:
    RAG_CHROMA_DIR  — ChromaDB persist directory
    RAG_DOCS_DIR    — Source document directory

Examples:
    python3 agent.py "boot process" --top 5
    python3 agent.py "chip errata" --type docs
    python3 agent.py --spec "ESP32-S3"
    python3 agent.py --compare "ESP32-S3,ESP32-C3"
    python3 agent.py --raw "CAN bus configuration"
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List

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

from main import ChromaIndex, query as rag_query, _format_comparison, SOC_TO_DATASHEET

# ═══════════════════════════════════════════════════════════════════════
# Source file reading (for deeper context when chunk text is insufficient)
# ═══════════════════════════════════════════════════════════════════════

def resolve_source_path(relpath: str) -> Path:
    docs_dir = os.environ['RAG_DOCS_DIR']
    candidate = Path(docs_dir) / relpath
    if candidate.exists():
        return candidate
    for f in Path(docs_dir).rglob(relpath):
        return f
    return candidate


def read_source_file(relpath: str, max_chars: int = 3000) -> str:
    fpath = resolve_source_path(relpath)
    if not fpath.exists():
        return ""
    if fpath.suffix.lower() in ('.pdf', '.zip', '.xlsx'):
        return ""
    try:
        text = fpath.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... (truncated)"
    return text


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

        src_content = read_source_file(source)
        if src_content:
            lines.append("> _Excerpt from source file:_")
            excerpt = re.sub(r'\n{3,}', '\n\n', src_content[:1500]).strip()
            lines.append(f"> {excerpt[:800]}")
            lines.append("")

    return '\n'.join(lines)


def do_spec_lookup(index, name: str) -> list:
    """Look up structured spec data for a named SoC."""
    name_normalized = name.strip().upper().replace('-', ' ').strip()
    # Try direct lookup first (name might match a doc_title)
    result = index.lookup_specs(name_normalized)
    if result:
        return result
    # Try via SOC_TO_DATASHEET mapping
    part_no = SOC_TO_DATASHEET.get(name_normalized)
    if part_no:
        result = index.lookup_specs(part_no)
        if result:
            return result
    # Fallback: check each datasheet prefix
    for soc_name, prefix in SOC_TO_DATASHEET.items():
        if soc_name in name_normalized or name_normalized.replace(' ', '') in soc_name.replace(' ', ''):
            result = index.lookup_specs(prefix)
            if result:
                return result
    return []


def do_comparison(index, names: list) -> list:
    """Compare two SoCs by name."""
    specs_list = []
    for name in names:
        specs_list.extend(do_spec_lookup(index, name.strip()))
    if len(specs_list) >= 2:
        comparison_table = _format_comparison(specs_list)
        if comparison_table:
            return [{'type': 'comparison', 'table': comparison_table, 'specs': specs_list}]
    return []

def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(__doc__.strip())
        sys.exit(0)

    raw = False
    top_n = 10
    doc_type = None
    spec_name = None
    compare_names = None
    query_parts = []

    i = 0
    while i < len(args):
        if args[i] == '--raw':
            raw = True
        elif args[i] == '--top' and i + 1 < len(args):
            top_n = int(args[i + 1])
            i += 1
        elif args[i] == '--type' and i + 1 < len(args):
            doc_type = args[i + 1]
            i += 1
        elif args[i] == '--spec' and i + 1 < len(args):
            spec_name = args[i + 1]
            i += 1
        elif args[i] == '--compare' and i + 1 < len(args):
            compare_names = [n.strip() for n in args[i + 1].split(',')]
            i += 1
        else:
            query_parts.append(args[i])
        i += 1

    query_text = ' '.join(query_parts).strip()

    chroma_dir = os.environ['RAG_CHROMA_DIR']
    index = ChromaIndex(chroma_dir)

    # Route: spec lookup
    if spec_name:
        specs = do_spec_lookup(index, spec_name)
        if specs and specs[0].get('specs'):
            spec = specs[0]
            lines = [f"## {spec['doc_title']} Specifications\n"]
            lines.append('| Feature | Value |')
            lines.append('|---------|-------|')
            for k, v in spec['specs'].items():
                lines.append(f"| {k} | {v} |")
            print('\n'.join(lines))
        else:
            print(f"No spec data found for '{spec_name}'")
        return

    # Route: comparison
    if compare_names and len(compare_names) >= 2:
        results = do_comparison(index, compare_names)
        if results:
            print(format_answer(results, raw=raw))
        else:
            print(f"No comparison data found for {compare_names}")
        return

    # Route: normal search
    if not query_text:
        print("Error: no query provided", file=sys.stderr)
        sys.exit(1)

    results = rag_query(index, query_text, n_results=top_n, doc_type=doc_type)
    print(format_answer(results, query_text, raw=raw))


if __name__ == '__main__':
    main()