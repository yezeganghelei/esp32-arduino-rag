#!/usr/bin/env python3
"""
ESP32 RAG — Gradio Web Chat Interface

A web-based chatbot UI for the ESP32 RAG knowledge base.
Connects to the FastAPI backend via HTTP.

Run:
    python -m scripts.web.app
    # or
    gradio scripts/web/app.py

Environment variables:
    API_BASE_URL   — backend API base URL (default: http://localhost:8000)
    WEB_HOST       — host to bind (default: 0.0.0.0)
    WEB_PORT       — port to bind (default: 7860)
"""

import os
import json
import uuid
import logging
from typing import List, Tuple, Generator, Optional

import gradio as gr
import httpx

# ── Monkey-patch: fix gradio_client json_schema bug ─────────────────
# Gradio 4.x crashes when parsing complex type annotations because
# schema can be a bool (True/False) instead of a dict.
try:
    from gradio_client import utils as gc_utils

    _orig_json_schema_to_python_type = gc_utils._json_schema_to_python_type

    def _patched_json_schema_to_python_type(schema, defs=None):
        try:
            return _orig_json_schema_to_python_type(schema, defs)
        except Exception:
            return "any"

    gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
except Exception:
    pass

logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "7860"))

DOC_TYPES = [
    ("全部", None),
    ("Datasheet 数据手册", "datasheet"),
    ("TRM 技术参考手册", "trm"),
    ("编程指南", "guide"),
    ("规格书", "specification"),
    ("API 参考", "api_reference"),
    ("安全手册", "safety"),
    ("发布说明", "release_notes"),
]

KB_KINDS = [
    ("🔍 自动识别 (文档+代码)", "auto"),
    ("📄 文档知识库", "docs"),
    ("💻 代码知识库", "code"),
]

# ── Helpers ─────────────────────────────────────────────────────────

def _api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _format_sources_md(sources: list) -> str:
    """Format source documents as a markdown reference list."""
    if not sources:
        return ""
    lines = ["\n\n---\n\n**📚 参考来源：**\n"]
    seen_titles = set()
    idx = 1
    for s in sources:
        title = s.get("doc_title", "Unknown")
        dtype = s.get("doc_type", "")
        score = s.get("score", 0)
        key = title
        if key in seen_titles:
            continue
        seen_titles.add(key)
        score_pct = f"{score*100:.1f}%" if score > 1 else f"{score:.4f}"
        lines.append(
            f"{idx}. **{title}** ({dtype}) — 相关度: {score_pct}"
        )
        idx += 1
    return "\n".join(lines)


# ── Chat logic (streaming) ──────────────────────────────────────────

def chat_stream(
    message: str,
    history: List[List[str]],
    session_id: str,
    doc_type: Optional[str],
    top_k: int,
    kb_kind: str,
) -> Generator[Tuple[str, List[List[str]], str], None, None]:
    """Stream chat response from the API.

    Yields: (current_assistant_message, updated_history, session_id)
    """
    if not message.strip():
        yield "", history, session_id
        return

    # Append user message to history
    history = history + [[message, ""]]
    yield "", history, session_id

    sources = []
    full_answer = []

    try:
        with httpx.stream(
            "POST",
            _api_url("/api/chat"),
            json={
                "query": message,
                "session_id": session_id,
                "doc_type": doc_type,
                "top_k": top_k,
                "use_history": True,
                "stream": True,
                "kind": kb_kind,
            },
            timeout=300.0,
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                error_msg = f"❌ 服务错误 ({response.status_code})"
                history[-1][1] = error_msg
                yield "", history, session_id
                return

            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "sources":
                    sources = event.get("data", [])

                elif event_type == "start":
                    data = event.get("data", {})
                    if data.get("session_id"):
                        session_id = data["session_id"]

                elif event_type == "delta":
                    content = event.get("content", "")
                    full_answer.append(content)
                    # Update the last assistant message
                    answer_text = "".join(full_answer)
                    history[-1][1] = answer_text
                    yield "", history, session_id

                elif event_type == "done":
                    # Add sources to the end of the answer
                    answer_text = "".join(full_answer)
                    sources_md = _format_sources_md(sources)
                    if sources_md:
                        answer_text += sources_md
                    history[-1][1] = answer_text
                    yield "", history, session_id

                elif event_type == "error":
                    error_msg = f"❌ {event.get('content', 'Unknown error')}"
                    history[-1][1] = error_msg
                    yield "", history, session_id
                    return

    except httpx.ConnectError:
        history[-1][1] = "❌ 无法连接到后端服务，请确认 API 服务已启动"
        yield "", history, session_id
    except Exception as e:
        history[-1][1] = f"❌ 出错了：{e}"
        yield "", history, session_id


def new_chat() -> Tuple[str, List, str]:
    """Start a new chat session."""
    new_id = _new_session_id()
    return "", [], new_id


# ── Gradio UI ───────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """Build the Gradio chat interface."""
    with gr.Blocks(
        title="ESP32 RAG 智能助手",
        theme=gr.themes.Soft(),
        css="""
        .chatbot { height: 70vh !important; }
        .sidebar { background: var(--color-background-secondary); }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🤖 ESP32 RAG 智能助手

            基于乐鑫官方文档和代码示例的语义检索问答系统，支持文档检索和代码检索。
            """
        )

        session_id = gr.State(value=_new_session_id())

        with gr.Row():
            # Sidebar
            with gr.Column(scale=1, elem_classes="sidebar"):
                gr.Markdown("### 📚 知识库")

                kb_kind_radio = gr.Radio(
                    choices=KB_KINDS,
                    value="auto",
                    label="选择知识库",
                    interactive=True,
                )

                gr.Markdown("### ⚙️ 设置")

                doc_type_dropdown = gr.Dropdown(
                    choices=DOC_TYPES,
                    value=None,
                    label="文档类型筛选",
                    interactive=True,
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=8,
                    step=1,
                    label="检索文档数量 (top_k)",
                    interactive=True,
                )

                gr.Markdown("### 📋 会话")
                new_chat_btn = gr.Button("🆕 新对话", variant="secondary")

                session_display = gr.Textbox(
                    value=lambda: session_id.value,
                    label="当前会话 ID",
                    interactive=False,
                    show_copy_button=True,
                )

                gr.Markdown("### 📊 状态")
                api_status = gr.Textbox(
                    value="检测中...",
                    label="后端连接",
                    interactive=False,
                )

            # Chat area
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="对话",
                    show_label=False,
                    height=700,
                    elem_classes="chatbot",
                    bubble_full_width=False,
                    show_copy_button=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你的问题，比如：ESP32-S3 的 GPIO 中断怎么配置？",
                        scale=5,
                        show_label=False,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        "ESP32-S3 的 GPIO 中断怎么配置？",
                        "ESP32 和 ESP32-S3 有什么区别？",
                        "介绍一下 ESP32-S3 的 Wi-Fi 性能",
                        "如何使用 Deep-sleep 低功耗模式？",
                    ],
                    inputs=msg_input,
                    label="💡 试试这些问题",
                )

        # ── Event handlers ────────────────────────────────────────

        msg_input.submit(
            fn=chat_stream,
            inputs=[msg_input, chatbot, session_id, doc_type_dropdown, top_k_slider, kb_kind_radio],
            outputs=[msg_input, chatbot, session_id],
            api_name=None,
        )

        send_btn.click(
            fn=chat_stream,
            inputs=[msg_input, chatbot, session_id, doc_type_dropdown, top_k_slider, kb_kind_radio],
            outputs=[msg_input, chatbot, session_id],
            api_name=None,
        )

        new_chat_btn.click(
            fn=new_chat,
            outputs=[msg_input, chatbot, session_id],
            api_name=None,
        )

        # Check API status on load
        async def check_api_status():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.get(_api_url("/health"))
                    if r.status_code == 200:
                        data = r.json()
                        return f"✅ 已连接 ({API_BASE_URL})"
            except Exception:
                pass
            return f"❌ 未连接 ({API_BASE_URL})"

        demo.load(fn=check_api_status, outputs=api_status, api_name=None)

    return demo


# ── Entry point ─────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting Gradio web UI on {WEB_HOST}:{WEB_PORT}")
    logger.info(f"Backend API: {API_BASE_URL}")

    demo = build_ui()
    demo.launch(
        server_name=WEB_HOST,
        server_port=WEB_PORT,
        share=False,
        show_api=False,
    )


if __name__ == "__main__":
    main()
