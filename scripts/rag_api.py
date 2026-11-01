#!/usr/bin/env python3
"""
Unified RAG query interface.

This is the SINGLE entry point for all query consumers:

  * scripts/agent.py       (CLI / skill)
  * scripts/api/server.py  (FastAPI web interface)
  * future MCP servers

All retrieval logic lives here and only here — the documentation + source-code
raw search and reciprocal-rank fusion — so every interface behaves identically.
Do not re-implement search in the callers; import `search` / `run_query` /
`multi_search` from this module. There is no spec/comparison detection; all
queries go through raw retrieval, and multi-entity questions are handled by the
decompose-and-search flow in the chat layer.

Modes:
  * "auto"  — docs DB (chroma_esp32_all) + code DB (chroma_esp32_code), merged
  * "docs"  — documentation DB only
  * "code"  — source-code DB only
"""

import os
import re
import sys
import threading
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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

from main import ChromaIndex, query as rag_query

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


# ── THE single search entry point ─────────────────────────────────────

def search(query: str, top_n: int = 10, doc_type: Optional[str] = None,
           mode: str = 'auto', code_query: Optional[str] = None) -> Tuple[list, str]:
    """Search the knowledge base.

    Args:
        query: search text (any language).
        top_n: number of results to return.
        doc_type: optional doc-type filter (datasheet, trm, guide, api, ...,
                  code). When set to anything other than None/"code", the code
                  DB is skipped (documentation-only), matching agent.py.
        mode: "auto" | "docs" | "code".
        code_query: optional code-oriented search text (used for the code DB in
                    "auto" mode; falls back to `query` when not given).

    Returns:
        (results, kind_used).

    NOTE: "auto" mode mirrors the skill's behaviour of running SEPARATE
    documentation and code searches (agent.py --docs-only + agent.py --code).
    The two result lists are CONCATENATED (docs first, then code) and each entry
    is tagged with its own doc_type — no reciprocal-rank fusion, so code results
    are not diluted by the documentation results.
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

    # ── auto: docs search + code search, concatenated (no RRF fusion) ──
    docs_index = get_doc_index()
    docs_results = rag_query(docs_index, query, n_results=top_n, doc_type=doc_type)

    # The code DB is ALWAYS searched in auto mode. `doc_type` is a DOCUMENTATION
    # type filter (datasheet/trm/guide/...), so it must not suppress the source
    # code search — otherwise selecting a doc-type filter in the web UI would
    # silently drop code results.
    code_results = []
    if has_code_index():
        try:
            code_docs, _ = _code_dirs()
            cq = code_query or query
            code_results = get_code_index().search(cq, n_results=top_n,
                                                   doc_type='code')
            for r in code_results:
                r['_src_root'] = code_docs
        except Exception:
            code_results = []

    return docs_results + code_results, 'auto'


# ── Routing: intent -> retrieval (always dual-search, no mode classification) ──

def run_query(route: dict, top_n: int = 10,
              doc_type: Optional[str] = None,
              forced_mode: Optional[str] = None) -> Tuple[list, str]:
    """Route a rewrite decision to retrieval.

    `route` is a dict with keys: docs_query, code_query, query.
    By default BOTH the docs DB and the code DB are searched (docs_query for
    docs, code_query for code), with the two result lists concatenated. This
    avoids relying on the LLM to classify "docs vs code" (which was the source
    of recurring errors where code/docs was dropped). `forced_mode` may be
    "docs" or "code" to restrict the search (the web UI's explicit selector).
    Returns (results, kind_used).
    """
    query = route.get("query") or ""

    if forced_mode == "code":
        return search(route.get("code_query") or query, top_n, doc_type, "code")

    if forced_mode == "docs":
        return search(route.get("docs_query") or query, top_n, doc_type, "docs")

    return search(route.get("docs_query") or query, top_n, doc_type, "auto",
                  route.get("code_query"))


# ── Multi-search (decompose -> search each sub-query in BOTH DBs) ──────

def multi_search(sub_queries: List[str], top_n: int = 10,
                 doc_type: Optional[str] = None,
                 mode: str = "auto") -> list:
    """Run several focused searches (one per sub-query) and aggregate.

    Each sub-query is a plain English phrase. `mode` controls which database(s)
    each sub-query searches:
      * "auto" -> docs DB + code DB (both, concatenated)
      * "docs" -> docs DB only
      * "code" -> code DB only
    This mirrors the KB selector so a "docs"/"code" selection stays strict and
    does not leak results from the other database during the decompose step.
    Results are deduplicated by chunk id.
    """
    docs_index = get_doc_index()
    code_index = None
    if has_code_index():
        try:
            code_index = get_code_index()
        except Exception:
            code_index = None

    merged: Dict[str, dict] = {}
    order: List[str] = []
    for sub in sub_queries:
        if not isinstance(sub, str):
            continue
        q = sub.strip()
        if not q:
            continue
        results = []
        if mode in ("auto", "docs"):
            try:
                results = docs_index.search(q, n_results=top_n, doc_type=doc_type)
            except Exception:
                results = []
        if mode in ("auto", "code") and code_index is not None:
            try:
                code_results = code_index.search(q, n_results=top_n, doc_type='code')
                for r in code_results:
                    r['_src_root'] = _code_dirs()[0]
                results = results + code_results
            except Exception:
                pass
        for r in results:
            if r.get('type') == 'comparison':
                continue
            rid = r.get('id') or (r.get('source') + '|' + r.get('doc_title') + '|' + r.get('text', '')[:40])
            if rid not in merged:
                merged[rid] = r
                order.append(rid)
    return [merged[rid] for rid in order]


# ── Source-file reading (for deeper context when chunk text is insufficient) ──

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
    """Read a source file referenced by a search result, if it is a text file."""
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
