#!/usr/bin/env python3
"""
LLM API wrapper — OpenAI-compatible chat completions.

Supports any provider with an OpenAI-compatible API:
  - DeepSeek
  - SiliconFlow (硅基流动)
  - OpenAI
  - vLLM / Ollama (OpenAI-compatible mode)
  - etc.

Configuration via environment variables:
  LLM_API_KEY      — API key (required)
  LLM_BASE_URL     — API base URL, e.g. https://api.deepseek.com/v1
  LLM_MODEL        — model name, e.g. deepseek-chat
  LLM_TEMPERATURE  — sampling temperature (default 0.3)
  LLM_MAX_TOKENS   — max tokens to generate (default 2048)
"""

import os
import json
import logging
from typing import List, Dict, Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


class LLMConfig:
    """Configuration loaded from environment variables."""

    def __init__(self):
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.model = os.environ.get("LLM_MODEL", "deepseek-chat")
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "2048"))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


def _build_messages(
    system_prompt: str,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the messages array for the chat completions API."""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


# ── Query rewriting (mimics the skill's LLM intent-understanding step) ──

REWRITE_SYSTEM_PROMPT = """You rewrite a user's question into TWO English search queries for an ESP32 documentation + source-code knowledge base.

Output a JSON object with exactly two keys:
- "docs": a natural-language query for the documentation search (concepts, architecture, configuration steps).
- "code": a code-oriented query for the source-code search (function names, register names, file names, keywords).

Rules:
- Output ONLY the JSON object. No explanation, no markdown fences, no trailing text.
- Translate any non-English content (e.g. Chinese) into English technical terms.
- "docs" should include technical synonyms (e.g. 时钟源 -> "clock source PLL XTAL crystal oscillator").
- "code" should use code identifiers/keywords (e.g. esp_clk, rtc_clk, clk_tree, xtal, pll, xclk, register).
- Keep chip/module/peripheral names exactly as-is (ESP32-S3, GPIO, I2S, UART, SPI, LEDC, etc.).
- Each query: 5-20 words.
"""


async def rewrite_queries(query: str, config: Optional[LLMConfig] = None) -> tuple:
    """Rewrite a user query (any language) into (docs_query, code_query) via the LLM.

    Falls back to the original query for both when the LLM is not configured or
    fails, so retrieval still works even without a rewrite.
    """
    if config is None:
        config = LLMConfig()

    if not config.is_configured:
        return query, query

    # Low temperature + short output is all we need for a keyword rewrite.
    rewrite_config = LLMConfig()
    rewrite_config.temperature = 0.0
    rewrite_config.max_tokens = 256

    try:
        answer = await chat_completion(
            system_prompt=REWRITE_SYSTEM_PROMPT,
            user_message=query,
            config=rewrite_config,
        )
    except Exception as e:
        logger.warning(f"Query rewrite failed, using raw query: {e}")
        return query, query

    docs_query = query
    code_query = query
    try:
        import re as _re
        m = _re.search(r'\{.*\}', answer, _re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            dq = (obj.get("docs") or "").strip()
            cq = (obj.get("code") or "").strip()
            if dq:
                docs_query = dq
            if cq:
                code_query = cq
    except Exception as e:
        logger.warning(f"Failed to parse rewrite JSON, using raw query: {e}")

    return docs_query, code_query


async def chat_completion(
    system_prompt: str,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    config: Optional[LLMConfig] = None,
) -> str:
    """Non-streaming chat completion. Returns the full response text."""
    import aiohttp

    if config is None:
        config = LLMConfig()

    if not config.is_configured:
        raise RuntimeError(
            "LLM is not configured. Set LLM_API_KEY and LLM_BASE_URL environment variables."
        )

    messages = _build_messages(system_prompt, user_message, history)
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }

    logger.debug(f"Calling LLM API: {config.model}, messages={len(messages)}")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"LLM API error {resp.status}: {text[:500]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


async def chat_completion_stream(
    system_prompt: str,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    config: Optional[LLMConfig] = None,
) -> AsyncIterator[str]:
    """Streaming chat completion. Yields text deltas as they arrive."""
    import aiohttp

    if config is None:
        config = LLMConfig()

    if not config.is_configured:
        raise RuntimeError(
            "LLM is not configured. Set LLM_API_KEY and LLM_BASE_URL environment variables."
        )

    messages = _build_messages(system_prompt, user_message, history)
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": True,
    }

    logger.debug(f"Calling LLM API (stream): {config.model}, messages={len(messages)}")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"LLM API error {resp.status}: {text[:500]}")

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
