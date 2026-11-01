#!/usr/bin/env python3
"""
RAG Chat — Retrieval-Augmented Generation for ESP32 documentation.

Combines the ChromaIndex search with an LLM to produce grounded answers
with source citations.

Usage:
    from scripts.api.chat import RAGChat
    chat = RAGChat(index)
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

# ── System prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位专业的 ESP32 技术支持工程师，基于乐鑫官方文档回答用户问题。

回答规则：
1. 严格基于提供的参考文档回答，不要编造信息
2. 如果参考文档中没有相关内容，明确告诉用户"在现有文档中未找到相关信息"，不要猜测
3. 回答要准确、专业、条理清晰
4. 技术细节（寄存器、地址、数值等）务必准确引用
5. 代码示例要完整可运行
6. 适当使用 Markdown 格式（标题、列表、代码块）组织回答
7. 在回答末尾列出引用的文档来源

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
    """RAG chat engine — retrieves docs, builds prompt, calls LLM."""

    def __init__(self, index=None, top_k: int = 8, retriever=None):
        self.index = index
        self.top_k = top_k
        self._retriever = retriever
        self._store = SessionStore()

    def get_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Get an existing session or create a new one."""
        if session_id:
            return self._store.get(session_id)
        return self._store.create()

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        self._store.delete(session_id)

    def _retrieve_sync(self, query: str, code_query: Optional[str] = None,
                       doc_type: Optional[str] = None) -> List[SourceDoc]:
        """Retrieve relevant documents from the index (or a custom retriever).

        Runs synchronously; call via _retrieve() so it executes off the
        event-loop thread and does not block other requests.
        """
        if self._retriever is not None:
            raw_results = self._retriever(query, code_query, doc_type, self.top_k)
        else:
            # Import here to avoid circular imports
            from main import query as rag_query
            raw_results = rag_query(
                self.index,
                query,
                n_results=self.top_k,
                doc_type=doc_type,
            )

        return self._to_sources(raw_results)

    async def _retrieve(self, query: str, code_query: Optional[str] = None,
                        doc_type: Optional[str] = None) -> List[SourceDoc]:
        """Retrieve relevant documents, running the blocking work in a thread."""
        return await asyncio.to_thread(self._retrieve_sync, query, code_query, doc_type)

    async def _rewrite_queries(self, query: str) -> Tuple[str, str]:
        """Rewrite the user query into (docs_query, code_query) via the LLM.

        Mirrors the skill's intent-understanding step: the raw query (often
        Chinese/colloquial) is translated/expanded into English technical terms
        for the documentation search AND into code-oriented terms for the
        source-code search. Falls back to the raw query when the LLM is missing
        or the rewrite fails.
        """
        from scripts.api.llm import rewrite_queries
        try:
            docs_query, code_query = await rewrite_queries(query)
        except Exception as e:
            logger.warning(f"Query rewrite failed, using raw query: {e}")
            return query, query
        return (docs_query or query), (code_query or query)

    @staticmethod
    def _to_sources(results) -> List[SourceDoc]:
        """Convert raw search result dicts to SourceDoc objects."""
        sources = []
        for r in results:
            if r.get('type') == 'comparison':
                continue
            sources.append(SourceDoc(
                doc_title=r.get('doc_title', ''),
                doc_type=r.get('doc_type', ''),
                text=r.get('text', ''),
                score=float(r.get('score', 0)),
                source_file=r.get('source', ''),
                section_path=r.get('section_path', ''),
            ))
        return sources

    def _build_context(self, sources: List[SourceDoc]) -> str:
        """Build the context block from retrieved sources."""
        parts = []
        for i, src in enumerate(sources, 1):
            # Truncate very long chunks
            text = src.text
            if len(text) > 2000:
                text = text[:2000] + "..."
            parts.append(
                f"[{i}] {src.doc_title} ({src.doc_type})\n"
                f"{text}"
            )
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

        # Rewrite the query (translate/expand to English tech terms for docs and
        # code), then retrieve with the rewritten queries. The original query is
        # kept for the user-facing prompt and history.
        docs_query, code_query = await self._rewrite_queries(query)
        sources = await self._retrieve(docs_query, code_query, doc_type)

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
            - type: "start" | "delta" | "sources" | "done" | "error"
            - content: text content (for delta)
            - data: additional data (for sources, done, etc.)
        """
        from scripts.api.llm import chat_completion_stream, LLMConfig

        try:
            session = self.get_session(session_id)

            # Rewrite query then retrieve with the rewritten queries.
            docs_query, code_query = await self._rewrite_queries(query)
            sources = await self._retrieve(docs_query, code_query, doc_type)

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
