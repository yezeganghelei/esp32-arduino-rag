#!/usr/bin/env python3
"""
RAG Chat — Retrieval-Augmented Generation for ESP32 documentation.

Combines the ChromaIndex search with an LLM to produce grounded answers
with source citations.

Usage:
    from scripts.api.chat import RAGChat
    chat = RAGChat(top_k=8)
    answer, sources = await chat.ask("How to configure GPIO interrupts?")
"""

import os
import re
import time
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# uvicorn only configures its own loggers, so this module's INFO logs would be
# silently dropped (propagating to an unconfigured root logger). Add a stderr
# handler so the retrieval stage timings are visible in `docker logs`.
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.DEBUG if os.environ.get("RAG_CHAT_DEBUG") == "1" else logging.INFO)
    logger.propagate = False

# ── System prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位专业的 ESP32 技术支持工程师，基于乐鑫官方文档和代码回答用户问题。

回答规则：
1. 严格基于提供的参考文档回答，绝不编造信息，不补充检索结果之外的事实。
2. 每个关键事实、数值、寄存器都必须标注来源，格式为 [Source: <文档标题>]。
3. 如果参考文档中没有相关内容，明确告诉用户"在现有文档中未找到相关信息"，不要猜测。
4. 如果多条参考文档互相矛盾，要分别指出每条来源的说法，说明冲突。
5. 回答准确、专业、条理清晰；技术细节（寄存器、地址、数值等）务必准确引用。
6. 适当使用 Markdown（标题、列表、代码块）组织回答。
7. 当用户明确要求某种实现方式（Arduino / MicroPython / ESP-IDF / IDF 等）时，只要参考文档中出现了该方式的代码（哪怕是驱动文件、BSP、片段、函数实现），就必须把它**完整地**放进代码块输出（不要只摘一两行、不要用文字总结代替代码），并标注来源文件（[Source: <文件名>]）。不要因为"不是完整的可运行示例"就宣称"未找到该方式"。
8. 在回答末尾列出引用的文档来源。

参考文档格式：
---
[编号] 文档标题 (文档类型)
内容片段
---
"""


# ── Data models ─────────────────────────────────────────────────────

@dataclass
class SourceDoc:
    """A single source document used in the answer."""
    doc_title: str
    doc_type: str
    text: str
    score: float
    source_file: str = ""
    section_path: str = ""
    src_root: str = ""
    chunk_id: str = ""


@dataclass
class ChatSession:
    """Conversation history for a single session."""
    session_id: str
    history: List[Dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.last_used = time.time()

    def clear(self):
        self.history.clear()


# ── Session store (in-memory) ───────────────────────────────────────

class SessionStore:
    """Simple in-memory session store with auto-cleanup of old sessions."""

    def __init__(self, max_sessions: int = 100, ttl_seconds: int = 3600):
        self._sessions: Dict[str, ChatSession] = {}
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds

    def get(self, session_id: str) -> ChatSession:
        """Get or create a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id=session_id)
        self._sessions[session_id].last_used = time.time()
        self._cleanup_old()
        return self._sessions[session_id]

    def create(self) -> ChatSession:
        """Create a new session with a generated ID."""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id=session_id)
        self._sessions[session_id] = session
        self._cleanup_old()
        return session

    def delete(self, session_id: str):
        """Delete a session."""
        self._sessions.pop(session_id, None)

    def _cleanup_old(self):
        """Remove expired sessions if we're over the limit."""
        if len(self._sessions) <= self._max_sessions:
            return
        now = time.time()
        # Remove expired first
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_used > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        # If still over limit, remove oldest by last_used
        if len(self._sessions) > self._max_sessions:
            sorted_sessions = sorted(
                self._sessions.values(), key=lambda s: s.last_used
            )
            to_remove = sorted_sessions[:len(self._sessions) - self._max_sessions]
            for s in to_remove:
                del self._sessions[s.session_id]


# ── RAG Chat core ───────────────────────────────────────────────────

class RAGChat:
    """RAG chat engine — understands intent, retrieves, builds prompt, calls LLM.

    Retrieval goes through `rag_api.run_query` (the same unified kernel used by
    the skill's agent.py), so the web and the skill share identical retrieval
    behaviour. Both the docs DB and the code DB are always searched (no
    "docs vs code" classification); `forced_mode` ("docs"/"code") is only set by
    the web UI's explicit selector to restrict the search.
    """

    def __init__(self, top_k: int = 8, forced_mode: Optional[str] = None):
        self.top_k = top_k
        self.forced_mode = forced_mode
        self._store = SessionStore()

    def get_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Get an existing session or create a new one."""
        if session_id:
            return self._store.get(session_id)
        return self._store.create()

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        self._store.delete(session_id)

    def _retrieve_sync(self, route: dict, doc_type: Optional[str] = None) -> List[SourceDoc]:
        """Retrieve relevant documents via rag_api.run_query.

        Runs synchronously; call via _retrieve() so it executes off the
        event-loop thread and does not block other requests.
        """
        import rag_api
        t0 = time.time()
        raw_results, _kind = rag_api.run_query(route, self.top_k, doc_type,
                                               forced_mode=self.forced_mode)
        logger.info("retrieve: %.2fs | kind=%s results=%d",
                    time.time() - t0, _kind, len(raw_results))
        return self._to_sources(raw_results)

    async def _retrieve(self, route: dict, doc_type: Optional[str] = None) -> List[SourceDoc]:
        """Retrieve relevant documents, running the blocking work in a thread."""
        return await asyncio.to_thread(self._retrieve_sync, route, doc_type)

    async def _rewrite(self, query: str) -> dict:
        """Produce English search queries (docs_query + code_query) via the LLM.

        No "docs vs code" mode is decided here — both databases are always
        searched, so a wrong classification can never drop code or docs.
        Falls back to the raw query when the LLM is missing or fails.
        """
        from scripts.api.llm import rewrite_queries
        t0 = time.time()
        try:
            route = await rewrite_queries(query)
        except Exception as e:
            logger.warning(f"Query rewrite failed, using raw query: {e}")
            route = {}
        route = dict(route or {})
        route.setdefault("query", query)
        if not route.get("docs_query"):
            route["docs_query"] = query
        if not route.get("code_query"):
            route["code_query"] = query
        logger.info("rewrite: %.2fs | needs_decompose=%s entities=%s docs_query=%r",
                    time.time() - t0, route.get("needs_decompose"),
                    route.get("entities"), route.get("docs_query"))
        return route

    async def _judge_and_decompose(self, query: str, context: str) -> dict:
        """LLM-judge whether `context` suffices to answer `query`; if not,
        return decomposed sub-queries (the "dig deeper" step)."""
        from scripts.api.llm import judge_and_decompose
        t0 = time.time()
        try:
            result = await judge_and_decompose(query, context)
        except Exception as e:
            logger.warning(f"judge_and_decompose failed: {e}")
            result = {"sufficient": True}
        logger.info("judge: %.2fs | sufficient=%s sub_queries=%d",
                    time.time() - t0, result.get("sufficient"),
                    len(result.get("sub_queries", [])))
        return result

    async def _multi_search(self, sub_queries: List[str]) -> List[SourceDoc]:
        """Search docs + code for each sub-query (respecting forced_mode)."""
        import rag_api
        mode = self.forced_mode or "auto"
        logger.info("multi_search: %d sub-queries, mode=%s", len(sub_queries), mode)
        t0 = time.time()
        try:
            raw = await asyncio.to_thread(rag_api.multi_search, sub_queries, 10, None, mode)
        except Exception as e:
            logger.warning(f"multi_search failed: {e}")
            return []
        logger.info("multi_search: %.2fs | %d results",
                    time.time() - t0, len(raw))
        return self._to_sources(raw)

    @staticmethod
    def _merge_sources(existing: List[SourceDoc], extra: List[SourceDoc]) -> List[SourceDoc]:
        """Dedupe and append extra sources onto existing.

        Dedupes by chunk id when available, otherwise by (source, title,
        section), so different chunks of the SAME document are preserved (e.g.
        the TRM's "45 GPIO" chunk vs its "GPIO matrix" chunk).
        """
        seen = set()
        for s in existing:
            seen.add(s.chunk_id or (s.source_file, s.doc_title, s.section_path))
        merged = list(existing)
        for s in extra:
            key = s.chunk_id or (s.source_file, s.doc_title, s.section_path)
            if key not in seen:
                seen.add(key)
                merged.append(s)
        return merged

    async def _gather_sources(self, query: str, route: dict,
                              doc_type: Optional[str] = None) -> List[SourceDoc]:
        """Retrieve sources; when the query needs decomposition and the direct
        answer is insufficient, dig deeper by searching sub-queries.

        Mirrors the skill's "exhaust recall / re-search with different keywords"
        behaviour, but driven automatically by the LLM.
        """
        t0 = time.time()
        sources = await self._retrieve(route, doc_type)

        if route.get("needs_decompose"):
            direct_ctx = self._build_context(sources)
            decision = await self._judge_and_decompose(query, direct_ctx)
            if not decision.get("sufficient", True) and decision.get("sub_queries"):
                extra = await self._multi_search(decision["sub_queries"])
                sources = self._merge_sources(sources, extra)

        n_code = sum(1 for s in sources if s.doc_type == "code")
        logger.info("gather_sources: %.2fs total | %d sources (%d docs / %d code) | query: %s",
                    time.time() - t0, len(sources), len(sources) - n_code, n_code, query)
        return sources

    @staticmethod
    def _to_sources(results) -> List[SourceDoc]:
        """Convert raw search result dicts to SourceDoc objects.

        Converts regular chunk results (docs/code) to SourceDoc with text + source.
        """
        sources = []
        for r in results:
            sources.append(SourceDoc(
                doc_title=r.get('doc_title', ''),
                doc_type=r.get('doc_type', ''),
                text=r.get('text', ''),
                score=float(r.get('score', 0)),
                source_file=r.get('source', ''),
                section_path=r.get('section_path', ''),
                src_root=r.get('_src_root', ''),
                chunk_id=r.get('id', ''),
            ))
        return sources

    def _build_context(self, sources: List[SourceDoc]) -> str:
        """Build the context block from retrieved sources.

        When a source chunk is a text/code file, also reads the source file to
        append a deeper excerpt (mirrors the skill's source-file fallback).
        Caps the number of sources so the LLM context stays focused.
        """
        import rag_api
        max_sources = 60
        parts = []
        seen_code = set()
        for i, src in enumerate(sources):
            if len(parts) >= max_sources:
                break
            text = src.text
            if len(text) > 1500:
                text = text[:1500] + "..."
            # Collapse exact-duplicate source chunks (the same BSP driver copied
            # into many examples) so one framework's code is not buried by many
            # copies of another framework's code.
            key = (src.doc_title, text)
            if src.doc_type == "code" and key in seen_code:
                continue
            seen_code.add(key)
            block = f"[{i}] {src.doc_title} ({src.doc_type})\n{text}"
            if src.source_file:
                try:
                    excerpt = rag_api.read_source_file(
                        src.source_file, max_chars=1500,
                        extra_root=src.src_root or None)
                except Exception:
                    excerpt = ""
                if excerpt:
                    excerpt = re.sub(r'\n{3,}', '\n\n', excerpt).strip()
                    block += f"\n> _Source file excerpt:_\n{excerpt[:900]}"
            parts.append(block)
        return "\n\n---\n\n".join(parts)

    def _build_user_prompt(self, query: str, sources: List[SourceDoc]) -> str:
        """Build the full user prompt with context."""
        context = self._build_context(sources)
        return (
            f"参考文档：\n\n{context}\n\n"
            f"====================\n\n"
            f"用户问题：{query}\n\n"
            f"请基于以上参考文档回答用户的问题。如果参考文档中没有相关内容，请明确说明。"
        )

    async def ask(
        self,
        query: str,
        session_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        use_history: bool = True,
    ) -> Tuple[str, List[SourceDoc], str]:
        """Ask a question and get a full answer (non-streaming).

        Returns:
            (answer_text, sources_list, session_id)
        """
        from scripts.api.llm import chat_completion, LLMConfig

        session = self.get_session(session_id)

        # Understand intent (rewrite to mode + English queries), then retrieve
        # (with decompose-and-search fallback when no direct answer).
        route = await self._rewrite(query)
        sources = await self._gather_sources(query, route, doc_type)

        # Build prompt
        user_prompt = self._build_user_prompt(query, sources)
        history = session.history if use_history else None

        # Call LLM
        config = LLMConfig()
        answer = await chat_completion(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_prompt,
            history=history,
            config=config,
        )

        # Update session history
        session.add_message("user", query)
        session.add_message("assistant", answer)

        return answer, sources, session.session_id

    async def ask_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        use_history: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Ask a question with streaming response.

        Yields dicts with:
            - type: "start" | "delta" | "sources" | "done" | "error" | "progress"
            - content: text content (for delta / progress)
            - stage: progress stage name (for progress)
            - data: additional data (for sources, done, etc.)
        """
        from scripts.api.llm import chat_completion_stream, LLMConfig

        try:
            session = self.get_session(session_id)

            # Understand intent (rewrite to English queries).
            yield {"type": "progress", "stage": "rewrite", "content": "理解意图中…"}
            route = await self._rewrite(query)

            # Retrieve (with decompose-and-search fallback when no direct answer).
            yield {"type": "progress", "stage": "retrieve", "content": "检索文档和代码中…"}
            sources = await self._gather_sources(query, route, doc_type)

            # Send sources first
            sources_data = [
                {
                    "doc_title": s.doc_title,
                    "doc_type": s.doc_type,
                    "text": s.text[:500],  # truncated for streaming
                    "score": s.score,
                    "source_file": s.source_file,
                    "section_path": s.section_path,
                }
                for s in sources
            ]
            yield {"type": "sources", "data": sources_data}

            # Build prompt
            user_prompt = self._build_user_prompt(query, sources)
            history = session.history if use_history else None

            # Stream LLM response
            config = LLMConfig()
            full_answer = []
            yield {"type": "start", "data": {"session_id": session.session_id}}
            yield {"type": "progress", "stage": "answer", "content": "生成回答中…"}

            async for delta in chat_completion_stream(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_prompt,
                history=history,
                config=config,
            ):
                full_answer.append(delta)
                yield {"type": "delta", "content": delta}

            # Update session history with complete answer
            session.add_message("user", query)
            session.add_message("assistant", "".join(full_answer))

            yield {"type": "done", "data": {
                "session_id": session.session_id,
                "total_chars": len("".join(full_answer)),
            }}

        except Exception as e:
            logger.exception("Chat stream error")
            yield {"type": "error", "content": str(e)}


# ── Shared one-shot RAG answer (web / MCP / CLI use the SAME flow) ──

def format_rag_answer(answer: str, sources: List[SourceDoc]) -> str:
    """Format an answer plus its source list into readable text (MCP / CLI)."""
    parts = [answer]
    if sources:
        parts.append("\n---\n\n参考来源：")
        seen = set()
        idx = 1
        for s in sources:
            key = (s.doc_title, s.doc_type)
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"{idx}. {s.doc_title} ({s.doc_type})")
            idx += 1
    return "\n".join(parts)


async def rag_answer(query: str, top_k: int = 8,
                     doc_type: Optional[str] = None,
                     mode: str = "auto") -> Tuple[str, List[SourceDoc]]:
    """One-shot RAG answer: rewrite -> retrieve (docs+code) -> generate.

    This is the single shared flow used by the web chat, the MCP server and the
    CLI, so all three produce identical answers for the same query.

    mode: "auto" (docs+code) | "docs" | "code".
    Returns (answer_text, sources).
    """
    forced = None if mode == "auto" else mode
    chat = RAGChat(top_k=top_k, forced_mode=forced)
    answer, sources, _ = await chat.ask(query, doc_type=doc_type, use_history=False)
    return answer, sources
