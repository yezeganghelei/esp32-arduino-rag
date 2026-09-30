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
    --code             Query only the source-code database (config.yaml `code:`)
    --docs-only        Query only the documentation database

By default a normal query searches BOTH the documentation database and the
source-code database (when it exists) and merges the results.

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

from main import (ChromaIndex, query as rag_query, _format_comparison,
                  SOC_TO_DATASHEET)

# ═══════════════════════════════════════════════════════════════════════
# Source file reading (for deeper context when chunk text is insufficient)
# ═══════════════════════════════════════════════════════════════════════

def _source_roots() -> List[Path]:
    """Candidate source roots, most specific first.

    The DB stores *relative* paths, so the only thing that varies per machine
    is the source root. Try the configured RAG_DOCS_DIR first, then the
    portable defaults that ship inside the skill.
    """
    roots: List[Path] = []
    env = os.environ.get('RAG_DOCS_DIR')
    if env:
        roots.append(Path(env))
    roots.append(SKILL_DIR / 'source')
    roots.append(SKILL_DIR / 'source_code')
    seen = set()
    out: List[Path] = []
    for r in roots:
        key = str(r).lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def resolve_source_path(relpath: str, extra_root: Optional[str] = None) -> Path:
    # Normalize separators so a Windows-built index resolves on Linux too.
    parts = [p for p in re.split(r'[\\/]+', relpath) if p]
    roots = _source_roots()
    if extra_root:
        roots = [Path(extra_root)] + [r for r in roots
                                      if str(r).lower() != str(Path(extra_root)).lower()]
    for root in roots:
        if not parts:
            break
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    # Fallback: locate the file by name anywhere under the roots.
    if parts:
        name = parts[-1]
        for root in roots:
            if root.is_dir():
                try:
                    for f in root.rglob(name):
                        return f
                except Exception:
                    pass
    base = roots[0] if roots else Path('.')
    return base.joinpath(*parts) if parts else base


def read_source_file(relpath: str, max_chars: int = 3000,
                     extra_root: Optional[str] = None) -> str:
    fpath = resolve_source_path(relpath, extra_root)
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


def _code_dirs():
    """Return (code_docs_dir, code_chroma_dir) for the source-code DB.

    Read directly from config.yaml's `code:` block so it is unaffected by the
    RAG_DOCS_DIR default (which points at the documentation source).
    """
    cfg_path = SKILL_DIR / 'config.yaml'
    code = {}
    try:
        import yaml
        with open(cfg_path, encoding='utf-8') as f:
            code = (yaml.safe_load(f) or {}).get('code', {}) or {}
    except Exception:
        code = {}
    docs = code.get('docs_dir') or ''
    if docs and not os.path.isabs(docs):
        docs = str(SKILL_DIR / docs)
    chroma = code.get('chroma_dir') or '.chroma_esp32_code'
    if not os.path.isabs(chroma):
        chroma = str(SKILL_DIR / chroma)
    return docs, chroma


def _merge_results(docs_results: list, code_results: list, top_n: int) -> list:
    """Merge documentation and code result lists by reciprocal-rank fusion.

    Scores across two independent databases are not directly comparable, so we
    fuse on rank instead (same idea as the multi-collection RRF used inside the
    documentation DB).
    """
    if not code_results:
        return docs_results[:top_n]
    if not docs_results:
        return code_results[:top_n]
    merged = []
    for i, r in enumerate(docs_results):
        d = dict(r)
        d['_rrf'] = 1.0 / (60.0 + i)
        merged.append(d)
    for i, r in enumerate(code_results):
        d = dict(r)
        d['_rrf'] = 1.0 / (60.0 + i)
        merged.append(d)
    merged.sort(key=lambda x: -x['_rrf'])
    return merged[:top_n]


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

    docs_docs = os.environ.get('RAG_DOCS_DIR') or str(SKILL_DIR / 'source')
    docs_chroma = os.environ.get('RAG_CHROMA_DIR') or str(SKILL_DIR / '.chroma_esp32_all')
    code_docs, code_chroma = _code_dirs()

    # ── spec / comparison routes are documentation-only ──
    if spec_name or (compare_names and len(compare_names) >= 2):
        os.environ['RAG_CHROMA_DIR'] = docs_chroma
        os.environ['RAG_DOCS_DIR'] = docs_docs
        index = ChromaIndex(docs_chroma)

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

        results = do_comparison(index, compare_names)
        if results:
            print(format_answer(results, raw=raw))
        else:
            print(f"No comparison data found for {compare_names}")
        return

    if not query_text:
        print("Error: no query provided", file=sys.stderr)
        sys.exit(1)

    if code_mode:
        # source-code DB only
        if not code_chroma or not os.path.isdir(code_chroma):
            print("Error: source-code database not found", file=sys.stderr)
            sys.exit(1)
        os.environ['RAG_CHROMA_DIR'] = code_chroma
        os.environ['RAG_DOCS_DIR'] = code_docs
        index = ChromaIndex(code_chroma)
        if doc_type is None:
            doc_type = 'code'
        results = rag_query(index, query_text, n_results=top_n, doc_type=doc_type)
        for r in results:
            r['_src_root'] = code_docs
        print(format_answer(results, query_text, raw=raw))
        return

    if docs_only:
        # documentation DB only
        os.environ['RAG_CHROMA_DIR'] = docs_chroma
        os.environ['RAG_DOCS_DIR'] = docs_docs
        index = ChromaIndex(docs_chroma)
        results = rag_query(index, query_text, n_results=top_n, doc_type=doc_type)
        print(format_answer(results, query_text, raw=raw))
        return

    # ── default: dual search (documentation + code), merged by RRF ──
    os.environ['RAG_CHROMA_DIR'] = docs_chroma
    os.environ['RAG_DOCS_DIR'] = docs_docs
    docs_index = ChromaIndex(docs_chroma)
    docs_results = rag_query(docs_index, query_text, n_results=top_n, doc_type=doc_type)

    # an auto-detected comparison/spec query short-circuits to docs only
    if docs_results and docs_results[0].get('type') == 'comparison':
        print(format_answer(docs_results, query_text, raw=raw))
        return

    code_results = []
    if doc_type in (None, 'code') and code_chroma and os.path.isdir(code_chroma):
        try:
            code_index = ChromaIndex(code_chroma, embedding_function=docs_index._ef)
            code_results = code_index.search(query_text, n_results=top_n, doc_type='code')
        except Exception:
            code_results = []
        for r in code_results:
            r['_src_root'] = code_docs

    merged = _merge_results(docs_results, code_results, top_n)
    print(format_answer(merged, query_text, raw=raw))


if __name__ == '__main__':
    main()