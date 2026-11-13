#!/usr/bin/env python3
"""
ESP32 RAG — FastAPI Server

Provides HTTP API endpoints for the ESP32 RAG knowledge base.

Endpoints:
    GET  /health              — health check + basic stats
    POST /api/search          — search the documentation index
    GET  /api/stats           — index statistics

Run:
    uvicorn scripts.api.server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import asyncio
import threading
import logging
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── pysqlite3 patch (MUST be before any chromadb import) ──────────
try:
    import pysqlite3 as _pysqlite3
    sys.modules['sqlite3'] = _pysqlite3
except ImportError:
    pass

# ── Path setup ─────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).resolve().parent.parent.parent
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

# Silence noisy chromadb logs
logging.getLogger('chromadb').setLevel(logging.ERROR)
for _name in ['chromadb.segment.impl.metadata', 'chromadb.segment.impl.vector',
             'chromadb.telemetry', 'chromadb.rate_limiting']:
    logging.getLogger(_name).setLevel(logging.ERROR)

import rag_api

# ── Pydantic models ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_n: int = Field(10, ge=1, le=50, description="Number of results to return")
    doc_type: Optional[str] = Field(None, description="Filter by document type (datasheet, trm, guide, api, safety, release_notes, specification, docs, code)")
    hybrid: bool = Field(True, description="Enable BM25 hybrid retrieval")
    rerank: bool = Field(True, description="Enable cross-encoder reranking")
    rewrite: bool = Field(True, description="Rewrite the query into English search keywords via the LLM before retrieval")


class SearchResult(BaseModel):
    score: float
    doc_type: str
    source: str
    section_path: str
    doc_title: str
    text: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    elapsed_ms: float


class StatsResponse(BaseModel):
    chroma_dir: str
    docs_dir: str
    total_chunks: int
    total_docs: int
    collections: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    doc_index_loaded: bool
    code_index_loaded: bool
    code_index_available: bool
    model_name: str
    chroma_dir: str
    code_chroma_dir: str


# ── App initialization ─────────────────────────────────────────────

app = FastAPI(
    title="ESP32 RAG API",
    description="ESP32 documentation knowledge base — semantic search API (ChromaDB + BM25 + cross-encoder)",
    version="1.0.0",
)

# ── Helper: normalize result dicts ─────────────────────────────────

def _result_to_dict(r: dict) -> dict:
    """Normalize a search result dict to our API schema."""
    return {
        'score': float(r.get('score', 0)),
        'doc_type': r.get('doc_type', ''),
        'source': r.get('source', ''),
        'section_path': r.get('section_path', ''),
        'doc_title': r.get('doc_title', ''),
        'text': r.get('text', ''),
    }


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    doc_loaded = rag_api.doc_index_loaded()
    code_loaded = rag_api.code_index_loaded()
    return {
        'status': 'ok',
        'index_loaded': doc_loaded or code_loaded,
        'doc_index_loaded': doc_loaded,
        'code_index_loaded': code_loaded,
        'code_index_available': rag_api.has_code_index(),
        'model_name': rag_api.current_model_name() or "—",
        'chroma_dir': os.environ.get('RAG_CHROMA_DIR', ''),
        'code_chroma_dir': os.environ.get('RAG_CODE_CHROMA_DIR', ''),
    }


@app.get("/api/stats", tags=["System"])
async def stats(kind: str = Query("docs", description="Index to query: 'docs' or 'code'")):
    """Get index statistics (chunk counts per collection)."""
    if kind not in ("docs", "code"):
        raise HTTPException(status_code=400, detail="kind must be 'docs' or 'code'")

    index = rag_api.get_doc_index() if kind == "docs" else rag_api.get_code_index()
    collections: Dict[str, int] = {}
    try:
        collections['chunks'] = index.chunks.count()
        collections['docs'] = index.docs.count()
        collections['specs'] = index.specs.count()
        for dtype, col in index._collections.items():
            try:
                collections[dtype] = col.count()
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")

    total_chunks = collections.get('chunks', 0)
    total_docs = collections.get('docs', 0)

    return {
        'kind': kind,
        'chroma_dir': index._chroma_dir,
        'total_chunks': total_chunks,
        'total_docs': total_docs,
        'collections': collections,
    }


@app.post("/api/search", tags=["Search"])
async def search(req: SearchRequest, kind: str = Query("auto", description="Index to search: 'auto' (docs+code merged), 'docs' or 'code'")):
    """Search the ESP32 documentation and/or code index.

    `kind=auto` follows the skill's default query method: it auto-identifies the
    customer's intent and queries both chroma_esp32_all (docs) and
    chroma_esp32_code (code), merging results by reciprocal-rank fusion.

    Performs a three-stage retrieval:
    1. Dense vector search (sentence-transformers embedding)
    2. BM25 keyword hybrid fusion
    3. Cross-encoder reranking
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if kind not in ("auto", "docs", "code"):
        raise HTTPException(status_code=400, detail="kind must be 'auto', 'docs' or 'code'")

    start = time.time()

    # The search endpoint is the raw-search analog of agent.py's default search
    # (kind = auto/docs/code). The rewrite only translates the query; it does
    # not change the search kind here.
    docs_query = req.query
    code_query = req.query
    if req.rewrite:
        try:
            from scripts.api.llm import rewrite_queries
            route = await rewrite_queries(req.query)
            docs_query = route.get("docs_query") or req.query
            code_query = route.get("code_query") or req.query
        except Exception:
            docs_query = req.query
            code_query = req.query

    try:
        results, used_kind = await asyncio.to_thread(
            rag_api.search, docs_query, req.top_n, req.doc_type, kind, code_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    normalized = [_result_to_dict(r) for r in results]

    return {
        'query': req.query,
        'rewritten_docs_query': docs_query if docs_query != req.query else None,
        'rewritten_code_query': code_query if code_query != req.query else None,
        'kind': used_kind,
        'results': normalized,
        'total': len(normalized),
        'elapsed_ms': round((time.time() - start) * 1000, 2),
    }


# ── Chat / RAG endpoints ────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    session_id: Optional[str] = Field(None, description="Conversation session ID (for multi-turn chat)")
    doc_type: Optional[str] = Field(None, description="Filter by document type")
    top_k: int = Field(8, ge=1, le=20, description="Number of document chunks to retrieve")
    use_history: bool = Field(True, description="Use conversation history")
    stream: bool = Field(False, description="Stream the response (SSE)")
    kind: str = Field("auto", description="Knowledge base to use: 'auto' (docs+code merged), 'docs' or 'code'")
    language: str = Field("auto", description="Answer language: 'auto' (same as question), 'en', 'zh' or 'de'")


class ChatSource(BaseModel):
    doc_title: str
    doc_type: str
    text: str
    score: float
    source_file: str
    section_path: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]
    session_id: str
    elapsed_ms: float


class ChatSessionResponse(BaseModel):
    session_id: str
    message_count: int
    created_at: float
    last_used: float


_chat_engines: Dict[str, object] = {}  # kind -> RAGChat instance


def get_chat(kind: str = "auto"):
    """Get or create a RAGChat instance for the given query kind.

    kind: "auto" (LLM decides mode) | "docs" (force docs) | "code" (force code).
    """
    global _chat_engines
    if kind not in _chat_engines:
        from scripts.api.chat import RAGChat
        forced = None if kind == "auto" else kind
        _chat_engines[kind] = RAGChat(top_k=8, forced_mode=forced)
    return _chat_engines[kind]


@app.post("/api/chat", tags=["Chat"])
async def chat(req: ChatRequest):
    """Chat with the ESP32 RAG assistant (non-streaming).

    Returns a complete answer with source citations.
    Use `stream: true` for streaming SSE response.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if req.kind not in ("auto", "docs", "code"):
        raise HTTPException(status_code=400, detail="kind must be 'auto', 'docs' or 'code'")

    # For streaming requests, return SSE
    if req.stream:
        return StreamingResponse(
            _chat_stream(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    chat_engine = get_chat(req.kind)
    start = time.time()

    try:
        answer, sources, session_id = await chat_engine.ask(
            query=req.query,
            session_id=req.session_id,
            doc_type=req.doc_type,
            use_history=req.use_history,
            language=req.language,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger = logging.getLogger("uvicorn.error")
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    elapsed = (time.time() - start) * 1000

    return {
        'answer': answer,
        'sources': [
            {
                'doc_title': s.doc_title,
                'doc_type': s.doc_type,
                'text': s.text,
                'score': s.score,
                'source_file': s.source_file,
                'section_path': s.section_path,
            }
            for s in sources
        ],
        'session_id': session_id,
        'kind': req.kind,
        'elapsed_ms': round(elapsed, 2),
    }


async def _chat_stream(req: ChatRequest):
    """Streaming chat response generator (SSE format)."""
    import json
    chat_engine = get_chat(req.kind)

    async for event in chat_engine.ask_stream(
        query=req.query,
        session_id=req.session_id,
        doc_type=req.doc_type,
        use_history=req.use_history,
        language=req.language,
    ):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/api/chat/session/{session_id}", tags=["Chat"])
async def get_session_info(session_id: str, kind: str = Query("docs", description="Index kind: 'docs' or 'code'")):
    """Get information about a chat session (history length, etc.)."""
    chat_engine = get_chat(kind)
    session = chat_engine.get_session(session_id)
    return {
        'session_id': session.session_id,
        'message_count': len(session.history),
        'created_at': session.created_at,
        'last_used': session.last_used,
        'kind': kind,
    }


@app.delete("/api/chat/session/{session_id}", tags=["Chat"])
async def clear_session(session_id: str, kind: str = Query("docs", description="Index kind: 'docs' or 'code'")):
    """Clear conversation history for a session."""
    chat_engine = get_chat(kind)
    chat_engine.clear_session(session_id)
    return {"status": "ok", "session_id": session_id, "kind": kind}


# ── Startup event ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Log startup info and start background model warm-up."""
    logger = logging.getLogger("uvicorn.error")
    logger.info("=" * 60)
    logger.info("ESP32 RAG API Server starting...")
    logger.info(f"SKILL_DIR:   {SKILL_DIR}")
    logger.info(f"CHROMA_DIR (docs):  {os.environ.get('RAG_CHROMA_DIR')}")
    logger.info(f"DOCS_DIR (docs):    {os.environ.get('RAG_DOCS_DIR')}")
    logger.info(f"CHROMA_DIR (code):  {os.environ.get('RAG_CODE_CHROMA_DIR')} (available: {rag_api.has_code_index()})")
    logger.info(f"DOCS_DIR (code):    {os.environ.get('RAG_CODE_DOCS_DIR')}")
    logger.info("Warming up models and indexes in the background...")
    logger.info("=" * 60)
    threading.Thread(target=rag_api.warmup, daemon=True, name="rag-warmup").start()
