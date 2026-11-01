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
  LLM_TEMPERATURE  — sampling temperature (default 0.0)
  LLM_MAX_TOKENS   — max tokens to generate (default 4096)
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
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

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

REWRITE_SYSTEM_PROMPT = """You are the intent-understanding front-end of an ESP32 knowledge base. Rewrite the user's question into two English search queries for the raw retrieval engine.

Output a JSON object with these keys:
- "docs_query": a natural-language query for the documentation search (5-20 words).
- "code_query": a code-oriented query for the source-code search (function/register/file names + keywords, 5-20 words).
- "entities": a JSON array of the chip/module/peripheral names the question is asking about (e.g. ["esp32-s3", "esp32-wroom", "arduino-uno"]).
- "needs_decompose": boolean. true when the question must be decomposed into per-entity or per-method sub-queries (e.g. multi-entity comparison "compare the X of A, B, C", or multi-method "give IDF/Arduino/MicroPython methods"); false otherwise.

Rules:
- Output ONLY the JSON object. No explanation, no markdown fences, no trailing text.
- Translate any non-English content (e.g. Chinese) into English technical terms.
- "docs_query" should include technical synonyms (e.g. 时钟源 -> "clock source PLL XTAL crystal oscillator").
- "code_query" should use code identifiers/keywords (e.g. esp_clk, rtc_clk, clk_tree, xtal, pll, xclk, register).
- Keep chip/module/peripheral names exactly as-is (ESP32-S3, GPIO, I2S, UART, SPI, LEDC, etc.).
"""


async def rewrite_queries(query: str, config: Optional[LLMConfig] = None) -> dict:
    """Rewrite a user query (any language) into search queries via the LLM.

    Returns a dict with keys: docs_query, code_query, entities, needs_decompose,
    query. No "docs vs code" mode is decided here — the caller always searches
    both databases (docs_query for docs, code_query for code) and the results
    carry their own doc_type, so a wrong mode classification can never drop
    code or docs. Falls back to the raw query when the LLM fails.
    """
    fallback = {
        "docs_query": query,
        "code_query": query,
        "query": query,
        "entities": [],
        "needs_decompose": False,
    }
    if config is None:
        config = LLMConfig()

    if not config.is_configured:
        return fallback

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
        return fallback

    route = dict(fallback)
    try:
        import re as _re
        m = _re.search(r'\{.*\}', answer, _re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            dq = (obj.get("docs_query") or obj.get("docs") or "").strip()
            cq = (obj.get("code_query") or obj.get("code") or "").strip()
            if dq:
                route["docs_query"] = dq
            if cq:
                route["code_query"] = cq
            ent = obj.get("entities")
            if isinstance(ent, list):
                route["entities"] = [str(e).strip() for e in ent if str(e).strip()]
            route["needs_decompose"] = bool(obj.get("needs_decompose", False))
    except Exception as e:
        logger.warning(f"Failed to parse rewrite JSON, using raw query: {e}")

    return route


JUDGE_DECOMPOSE_SYSTEM_PROMPT = """You decide whether retrieved content is sufficient to answer a user's question, and if not, decompose the question into focused sub-queries.

Given the user's question and the retrieved content, output a JSON object:
- If the content fully answers the question (covers all entities and the asked dimension): {"sufficient": true}
- Otherwise: {"sufficient": false, "sub_queries": ["...", "..."]}

Rules for sub_queries:
- Output 1-5 English search queries, each targeting ONE missing entity/dimension/method.
- Phrase each sub-query to match the missing content: use code identifiers/function names for implementation questions (e.g. "spilcd.cpp lcd_init SPI2 Arduino", "machine.SPI MicroPython LCD example", "spi_bus_initialize SPI2_HOST esp_lcd"), and natural-language technical terms for factual questions (e.g. "ESP32-S3 number of GPIO pins total").
- For a COUNT/QUANTITY fact (number of GPIO/UART/SPI/I2C/PWM/ADC, pin count), phrase the query around the number itself, NOT the generic interface name or "datasheet". For example "ESP32-S3 number of GPIO pins total" / "how many UART does ESP32-S3 have" / "ESP32-S3 GPIO pin count 45".
- Avoid the word "datasheet" (counts often live in the Technical Reference Manual, not the datasheet).
- Each query: 5-15 words.
- Output ONLY the JSON object. No explanation, no markdown fences.
"""


async def judge_and_decompose(query: str, context: str,
                              config: Optional[LLMConfig] = None) -> dict:
    """Judge whether retrieved content suffices to answer `query`; if not, return
    decomposed sub-queries (the "dig deeper" step when there is no direct answer).
    """
    if config is None:
        config = LLMConfig()
    if not config.is_configured:
        return {"sufficient": True}

    rewrite_config = LLMConfig()
    rewrite_config.temperature = 0.0
    rewrite_config.max_tokens = 300

    user_message = f"用户问题：{query}\n\n检索到的内容：\n{context[:4000]}"
    try:
        answer = await chat_completion(
            system_prompt=JUDGE_DECOMPOSE_SYSTEM_PROMPT,
            user_message=user_message,
            config=rewrite_config,
        )
    except Exception as e:
        logger.warning(f"judge_and_decompose failed: {e}")
        return {"sufficient": True}

    try:
        import re as _re
        m = _re.search(r'\{.*\}', answer, _re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            sufficient = bool(obj.get("sufficient", True))
            subs = obj.get("sub_queries")
            out_subs = []
            if isinstance(subs, list):
                for s in subs:
                    if isinstance(s, dict):
                        q = str(s.get("query") or "").strip()
                    else:
                        q = str(s).strip()
                    if q:
                        out_subs.append(q)
            return {"sufficient": sufficient, "sub_queries": out_subs}
    except Exception as e:
        logger.warning(f"Failed to parse judge_and_decompose JSON: {e}")

    return {"sufficient": True}


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
