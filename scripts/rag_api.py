#!/usr/bin/env python3
"""
Unified RAG query interface.

This is the SINGLE entry point for all query consumers:

  * scripts/agent.py       (CLI / skill)
  * scripts/api/server.py  (FastAPI web interface)
  * future MCP servers

All retrieval logic lives here and only here — spec/comparison detection,
the documentation + source-code dual search, and reciprocal-rank fusion —
so every interface behaves identically. Do not re-implement search in the
callers; import `search` / `lookup_spec` / `compare` from this module.

Modes (same as the skill's agent.py):
  * "auto"  — docs DB (chroma_esp32_all) + code DB (chroma_esp32_code), merged
  * "docs"  — documentation DB only
  * "code"  — source-code DB only
"""

import os
import sys
import threading
import logging
from pathlib import Path
from typing import List, Optional, Tuple

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

# Default paths (can be overridden by env vars)
os.environ.setdefault('RAG_CHROMA_DIR', str(SKILL_DIR / '.chroma_esp32_all'))
os.environ.setdefault('RAG_DOCS_DIR', str(SKILL_DIR / 'source'))
os.environ.setdefault('RAG_CODE_CHROMA_DIR', str(SKILL_DIR / '.chroma_esp32_code'))
os.environ.setdefault('RAG_CODE_DOCS_DIR', str(SKILL_DIR / 'source_code'))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

logging.getLogger('chromadb').setLevel(logging.ERROR)
for _name in ['chromadb.segment.impl.metadata', 'chromadb.segment.impl.vector',
              'chromadb.telemetry', 'chromadb.rate_limiting']:
    logging.getLogger(_name).setLevel(logging.ERROR)

from main import (ChromaIndex, query as rag_query, _format_comparison,
                  SOC_TO_DATASHEET)

logger = logging.getLogger("rag_api")

# ── Index singletons (documentation + source code) ───────────────────
_doc_index: Optional[ChromaIndex] = None
_code_index: Optional[ChromaIndex] = None
_index_lock = threading.RLock()


def _code_dirs() -> Tuple[str, str]:
    """Return (code_docs_dir, code_chroma_dir) from config.yaml `code:` block."""
    cfg_path = SKILL_DIR / 'config.yaml'
    code = {}
    try:
        import yaml
        with open(cfg_path, encoding='utf-8') as f:
            code = (yaml.safe_load(f) or {}).get('code', {}) or {}
    except Exception:
        code = {}
    docs = code.get('docs_dir') or 'source_code'
    if docs and not os.path.isabs(docs):
        docs = str(SKILL_DIR / docs)
    chroma = code.get('chroma_dir') or '.chroma_esp32_code'
    if not os.path.isabs(chroma):
        chroma = str(SKILL_DIR / chroma)
    return docs, chroma


def get_doc_index() -> ChromaIndex:
    """Get/create the documentation index singleton (thread-safe)."""
    global _doc_index
    with _index_lock:
        if _doc_index is None:
            chroma_dir = os.environ.get(
                'RAG_CHROMA_DIR', str(SKILL_DIR / '.chroma_esp32_all'))
            if not os.path.isdir(chroma_dir):
                raise RuntimeError(
                    f"ChromaDB directory not found: {chroma_dir}. "
                    f"Set RAG_CHROMA_DIR or mount the docs vector DB volume.")
            _doc_index = ChromaIndex(chroma_dir)
    return _doc_index


def get_code_index() -> ChromaIndex:
    """Get/create the code index singleton, sharing the docs embedding function."""
    global _code_index
    with _index_lock:
        if _code_index is None:
            chroma_dir = os.environ.get(
                'RAG_CODE_CHROMA_DIR', str(SKILL_DIR / '.chroma_esp32_code'))
            if not os.path.isdir(chroma_dir):
                raise RuntimeError(
                    f"Code ChromaDB directory not found: {chroma_dir}. "
                    f"Set RAG_CODE_CHROMA_DIR or mount the code vector DB volume.")
            _code_index = ChromaIndex(
                chroma_dir, embedding_function=get_doc_index()._ef)
    return _code_index


def has_code_index() -> bool:
    """Whether the source-code DB directory exists."""
    chroma_dir = os.environ.get(
        'RAG_CODE_CHROMA_DIR', str(SKILL_DIR / '.chroma_esp32_code'))
    return os.path.isdir(chroma_dir)


def doc_index_loaded() -> bool:
    """Whether the documentation index singleton is already loaded."""
    return _doc_index is not None


def code_index_loaded() -> bool:
    """Whether the code index singleton is already loaded."""
    return _code_index is not None


def current_model_name() -> Optional[str]:
    """Return the loaded embedding model name, or None if not loaded yet."""
    if _doc_index is not None:
        return getattr(_doc_index._ef, '_model_name', 'unknown')
    if _code_index is not None:
        return getattr(_code_index._ef, '_model_name', 'unknown')
    return None


def warmup():
    """Preload models and force HNSW index load, to run in a background thread.

    The dense embedding model, cross-encoder, BM25 index and the code DB's
    HNSW vectors are expensive to load (~60s+ on CPU). Loading them eagerly
    avoids the first user request appearing to hang.
    """
    docs_index = get_doc_index()
    docs_index._ensure_bm25()
    docs_index.search("__warmup__", n_results=1, rerank=False)
    if has_code_index():
        get_code_index().search("__warmup__", n_results=1, doc_type='code',
                                rerank=False)
    from main import _get_reranker
    _get_reranker()
    logger.info("Model + index warm-up complete.")


# ── Reciprocal-rank fusion (docs + code) ──────────────────────────────

def _merge_results(docs_results: list, code_results: list, top_n: int) -> list:
    """Merge documentation and code result lists by reciprocal-rank fusion.

    Scores across two independent databases are not comparable, so fuse on
    rank (same approach as the multi-collection RRF inside main.py).
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


# ── THE single search entry point ─────────────────────────────────────

def search(query: str, top_n: int = 10, doc_type: Optional[str] = None,
           mode: str = 'auto') -> Tuple[list, str]:
    """Search the knowledge base.

    Args:
        query: search text (any language).
        top_n: number of results to return.
        doc_type: optional doc-type filter (datasheet, trm, guide, api, ...,
                  code). When set to anything other than None/"code", the code
                  DB is skipped (documentation-only), matching agent.py.
        mode: "auto" | "docs" | "code".

    Returns:
        (results, kind_used). kind_used is "docs" when a spec/comparison query
        short-circuited to documentation-only, otherwise equals `mode`.
    """
    if mode == 'docs':
        results = rag_query(get_doc_index(), query, n_results=top_n,
                            doc_type=doc_type)
        return results, 'docs'

    if mode == 'code':
        code_docs, _ = _code_dirs()
        results = get_code_index().search(query, n_results=top_n, doc_type='code')
        for r in results:
            r['_src_root'] = code_docs
        return results, 'code'

    # ── auto: docs + code, merged ──
    docs_index = get_doc_index()
    docs_results = rag_query(docs_index, query, n_results=top_n, doc_type=doc_type)

    # An auto-detected spec/comparison query is documentation-only.
    if docs_results and docs_results[0].get('type') == 'comparison':
        return docs_results, 'docs'

    code_results = []
    if doc_type in (None, 'code') and has_code_index():
        try:
            code_docs, _ = _code_dirs()
            code_results = get_code_index().search(query, n_results=top_n,
                                                   doc_type='code')
            for r in code_results:
                r['_src_root'] = code_docs
        except Exception:
            code_results = []

    return _merge_results(docs_results, code_results, top_n), 'auto'


# ── Structured spec lookup / comparison ──────────────────────────────

def lookup_spec(name: str) -> list:
    """Look up structured spec data for a named SoC (docs DB)."""
    name_normalized = name.strip().upper().replace('-', ' ').strip()
    index = get_doc_index()
    result = index.lookup_specs(name_normalized)
    if result:
        return result
    part_no = SOC_TO_DATASHEET.get(name_normalized)
    if part_no:
        result = index.lookup_specs(part_no)
        if result:
            return result
    for soc_name, prefix in SOC_TO_DATASHEET.items():
        if (soc_name in name_normalized
                or name_normalized.replace(' ', '') in soc_name.replace(' ', '')):
            result = index.lookup_specs(prefix)
            if result:
                return result
    return []


def compare(names: List[str]) -> list:
    """Compare specs of two or more SoCs by name."""
    specs_list = []
    for name in names:
        specs_list.extend(lookup_spec(name.strip()))
    if len(specs_list) >= 2:
        comparison_table = _format_comparison(specs_list)
        if comparison_table:
            return [{'type': 'comparison', 'table': comparison_table,
                     'specs': specs_list}]
    return []
