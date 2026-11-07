#!/usr/bin/env python3
"""
ESP32 RAG — Gradio Web Chat Interface

A web-based chatbot UI for the ESP32 RAG knowledge base.
Connects to the FastAPI backend via HTTP.

UI languages: 中文 (zh) / English (en) / Deutsch (de), selectable in the UI.

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

# ═══════════════════════════════════════════════════════════════════════
# i18n — interface languages
# ═══════════════════════════════════════════════════════════════════════

LANGS = [("中文", "zh"), ("English", "en"), ("Deutsch", "de")]
DEFAULT_LANG = "zh"

I18N = {
    "zh": {
        "app_title": "ESP32 RAG 智能助手",
        "lang_label": "界面语言",
        "title": "# 🤖 ESP32 RAG 智能助手\n\n基于乐鑫官方文档和代码示例的语义检索问答系统，支持文档检索和代码检索。",
        "kb_title": "### 📚 知识库",
        "kb_label": "选择知识库",
        "kb_auto": "🔍 自动识别 (文档+代码)",
        "kb_docs": "📄 文档知识库",
        "kb_code": "💻 代码知识库",
        "settings": "### ⚙️ 设置",
        "doc_type_label": "文档类型筛选",
        "doc_type_all": "全部",
        "doc_type_datasheet": "Datasheet 数据手册",
        "doc_type_trm": "TRM 技术参考手册",
        "doc_type_guide": "编程指南",
        "doc_type_spec": "规格书",
        "doc_type_api": "API 参考",
        "doc_type_safety": "安全手册",
        "doc_type_release": "发布说明",
        "top_k_label": "检索文档数量 (top_k)",
        "session_title": "### 📋 会话",
        "new_chat": "🆕 新对话",
        "session_label": "当前会话 ID",
        "status_title": "### 📊 状态",
        "api_label": "后端连接",
        "chatbot_label": "对话",
        "placeholder": "输入你的问题，比如：ESP32-S3 的 GPIO 中断怎么配置？",
        "send": "发送",
        "examples_label": "💡 试试这些问题",
        "examples": [
            "ESP32-S3 的 GPIO 中断怎么配置？",
            "ESP32 和 ESP32-S3 有什么区别？",
            "介绍一下 ESP32-S3 的 Wi-Fi 性能",
            "如何使用 Deep-sleep 低功耗模式？",
        ],
        "sources_title": "**📚 参考来源：**",
        "relevance": "相关度",
        "api_checking": "检测中...",
        "api_connected": "✅ 已连接",
        "api_disconnected": "❌ 未连接",
        "progress_rewrite": "🔍 理解意图中…",
        "progress_retrieve": "📚 检索文档和代码中…",
        "progress_decompose": "🔎 深挖检索中…",
        "progress_answer": "✍️ 生成回答中…",
        "progress_done": "✅ 完成",
        "progress_error": "❌ 出错",
        "progress_idle": "",
        "err_service": "❌ 服务错误",
        "err_connect": "❌ 无法连接到后端服务，请确认 API 服务已启动",
        "err_generic": "❌ 出错了",
    },
    "en": {
        "app_title": "ESP32 RAG Assistant",
        "lang_label": "Interface language",
        "title": "# 🤖 ESP32 RAG Assistant\n\nSemantic retrieval Q&A over Espressif's official documentation and code examples, with document and code search.",
        "kb_title": "### 📚 Knowledge Base",
        "kb_label": "Knowledge base",
        "kb_auto": "🔍 Auto (docs + code)",
        "kb_docs": "📄 Documentation",
        "kb_code": "💻 Source code",
        "settings": "### ⚙️ Settings",
        "doc_type_label": "Document type filter",
        "doc_type_all": "All",
        "doc_type_datasheet": "Datasheet",
        "doc_type_trm": "TRM (Technical Reference Manual)",
        "doc_type_guide": "Programming Guide",
        "doc_type_spec": "Specification",
        "doc_type_api": "API Reference",
        "doc_type_safety": "Safety Manual",
        "doc_type_release": "Release Notes",
        "top_k_label": "Number of chunks (top_k)",
        "session_title": "### 📋 Session",
        "new_chat": "🆕 New chat",
        "session_label": "Current session ID",
        "status_title": "### 📊 Status",
        "api_label": "Backend connection",
        "chatbot_label": "Chat",
        "placeholder": "Ask a question, e.g. How to configure GPIO interrupts on ESP32-S3?",
        "send": "Send",
        "examples_label": "💡 Try these questions",
        "examples": [
            "How to configure GPIO interrupts on ESP32-S3?",
            "What is the difference between ESP32 and ESP32-S3?",
            "ESP32-S3 Wi-Fi performance",
            "How to use Deep-sleep low power mode?",
        ],
        "sources_title": "**📚 Sources:**",
        "relevance": "relevance",
        "api_checking": "Checking...",
        "api_connected": "✅ Connected",
        "api_disconnected": "❌ Disconnected",
        "progress_rewrite": "🔍 Understanding intent…",
        "progress_retrieve": "📚 Retrieving docs and code…",
        "progress_decompose": "🔎 Deep retrieval…",
        "progress_answer": "✍️ Generating answer…",
        "progress_done": "✅ Done",
        "progress_error": "❌ Error",
        "progress_idle": "",
        "err_service": "❌ Service error",
        "err_connect": "❌ Cannot connect to the backend. Make sure the API service is running.",
        "err_generic": "❌ Error",
    },
    "de": {
        "app_title": "ESP32 RAG-Assistent",
        "lang_label": "Oberflächensprache",
        "title": "# 🤖 ESP32 RAG-Assistent\n\nSemantische Such- und Antwortmaschine über offizielle Espressif-Dokumentation und Codebeispiele, mit Dokument- und Codesuche.",
        "kb_title": "### 📚 Wissensdatenbank",
        "kb_label": "Wissensdatenbank",
        "kb_auto": "🔍 Automatisch (Doku + Code)",
        "kb_docs": "📄 Dokumentation",
        "kb_code": "💻 Quellcode",
        "settings": "### ⚙️ Einstellungen",
        "doc_type_label": "Dokumenttyp-Filter",
        "doc_type_all": "Alle",
        "doc_type_datasheet": "Datenblatt",
        "doc_type_trm": "TRM (Technisches Referenzhandbuch)",
        "doc_type_guide": "Programmierhandbuch",
        "doc_type_spec": "Spezifikation",
        "doc_type_api": "API-Referenz",
        "doc_type_safety": "Sicherheitshandbuch",
        "doc_type_release": "Versionshinweise",
        "top_k_label": "Anzahl der Textabschnitte (top_k)",
        "session_title": "### 📋 Sitzung",
        "new_chat": "🆕 Neue Unterhaltung",
        "session_label": "Aktuelle Sitzungs-ID",
        "status_title": "### 📊 Status",
        "api_label": "Backend-Verbindung",
        "chatbot_label": "Unterhaltung",
        "placeholder": "Stellen Sie eine Frage, z. B.: Wie konfiguriert man GPIO-Interrupts beim ESP32-S3?",
        "send": "Senden",
        "examples_label": "💡 Probieren Sie diese Fragen",
        "examples": [
            "Wie konfiguriert man GPIO-Interrupts beim ESP32-S3?",
            "Was ist der Unterschied zwischen ESP32 und ESP32-S3?",
            "ESP32-S3 WLAN-Leistung",
            "Wie nutzt man den Deep-Sleep-Energiesparmodus?",
        ],
        "sources_title": "**📚 Quellen:**",
        "relevance": "Relevanz",
        "api_checking": "Wird geprüft...",
        "api_connected": "✅ Verbunden",
        "api_disconnected": "❌ Nicht verbunden",
        "progress_rewrite": "🔍 Absicht wird verstanden…",
        "progress_retrieve": "📚 Doku und Code werden abgerufen…",
        "progress_decompose": "🔎 Vertiefte Suche…",
        "progress_answer": "✍️ Antwort wird erstellt…",
        "progress_done": "✅ Fertig",
        "progress_error": "❌ Fehler",
        "progress_idle": "",
        "err_service": "❌ Dienstfehler",
        "err_connect": "❌ Keine Verbindung zum Backend. Stellen Sie sicher, dass der API-Dienst läuft.",
        "err_generic": "❌ Fehler",
    },
}


def _tr(lang: str, key: str) -> str:
    """Translate a UI key for the given language (falls back to default)."""
    table = I18N.get(lang) or I18N[DEFAULT_LANG]
    return table.get(key, I18N[DEFAULT_LANG].get(key, key))


# Doc type: (i18n key, filter value)
DOC_TYPE_KEYS = [
    ("doc_type_all", None),
    ("doc_type_datasheet", "datasheet"),
    ("doc_type_trm", "trm"),
    ("doc_type_guide", "guide"),
    ("doc_type_spec", "specification"),
    ("doc_type_api", "api_reference"),
    ("doc_type_safety", "safety"),
    ("doc_type_release", "release_notes"),
]

KB_KIND_KEYS = [
    ("kb_auto", "auto"),
    ("kb_docs", "docs"),
    ("kb_code", "code"),
]


def _doc_type_choices(lang: str) -> list:
    return [(_tr(lang, k), v) for k, v in DOC_TYPE_KEYS]


def _kb_choices(lang: str) -> list:
    return [(_tr(lang, k), v) for k, v in KB_KIND_KEYS]


# ── Helpers ─────────────────────────────────────────────────────────

def _api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _format_sources_md(sources: list, lang: str = DEFAULT_LANG) -> str:
    """Format source documents as a markdown reference list."""
    if not sources:
        return ""
    lines = [f"\n\n---\n\n{_tr(lang, 'sources_title')}\n"]
    seen_titles = set()
    idx = 1
    rel = _tr(lang, "relevance")
    for s in sources:
        title = s.get("doc_title", "Unknown")
        dtype = s.get("doc_type", "")
        score = s.get("score", 0)
        key = title
        if key in seen_titles:
            continue
        seen_titles.add(key)
        score_pct = f"{score*100:.1f}%" if score > 1 else f"{score:.4f}"
        lines.append(f"{idx}. **{title}** ({dtype}) — {rel}: {score_pct}")
        idx += 1
    return "\n".join(lines)


# ── Progress bar ────────────────────────────────────────────────────

PROGRESS_PCT = {
    "rewrite": 20,
    "retrieve": 55,
    "decompose": 75,
    "answer": 90,
    "done": 100,
    "error": 100,
    "idle": 0,
}


def _progress_html(stage: str, lang: str = DEFAULT_LANG) -> str:
    """Render a progress bar for the given stage."""
    pct = PROGRESS_PCT.get(stage, 50)
    label = _tr(lang, f"progress_{stage}") if stage else ""
    color = "#dc2626" if stage == "error" else "#2563eb"
    if not label:
        return ""
    return (
        f'<div style="padding:4px 2px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="font-size:0.85em;color:#374151;">{label}</span>'
        f'<span style="font-size:0.85em;color:#9ca3af;">{pct}%</span>'
        f'</div>'
        f'<div style="background:#e5e7eb;border-radius:6px;height:8px;overflow:hidden;">'
        f'<div style="background:{color};height:100%;width:{pct}%;transition:width .3s ease;"></div>'
        f'</div></div>'
    )


# ── Chat logic (streaming) ──────────────────────────────────────────

def chat_stream(
    message: str,
    history: List[List[str]],
    session_id: str,
    doc_type: Optional[str],
    top_k: int,
    kb_kind: str,
    lang: str = DEFAULT_LANG,
) -> Generator[Tuple[str, List[List[str]], str, str], None, None]:
    """Stream chat response from the API.

    Yields: (current_assistant_message, updated_history, session_id, progress_html)
    """
    if not message.strip():
        yield "", history, session_id, _progress_html("idle", lang)
        return

    # Append user message to history
    history = history + [[message, ""]]
    yield "", history, session_id, _progress_html("rewrite", lang)

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
                error_msg = f"{_tr(lang, 'err_service')} ({response.status_code})"
                history[-1][1] = error_msg
                yield "", history, session_id, _progress_html("error", lang)
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

                if event_type == "progress":
                    stage = event.get("stage", "")
                    yield "", history, session_id, _progress_html(stage, lang)

                elif event_type == "sources":
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
                    yield "", history, session_id, _progress_html("answer", lang)

                elif event_type == "done":
                    # Add sources to the end of the answer
                    answer_text = "".join(full_answer)
                    sources_md = _format_sources_md(sources, lang)
                    if sources_md:
                        answer_text += sources_md
                    history[-1][1] = answer_text
                    yield "", history, session_id, _progress_html("done", lang)

                elif event_type == "error":
                    error_msg = f"❌ {event.get('content', 'Unknown error')}"
                    history[-1][1] = error_msg
                    yield "", history, session_id, _progress_html("error", lang)
                    return

    except httpx.ConnectError:
        history[-1][1] = _tr(lang, "err_connect")
        yield "", history, session_id, _progress_html("error", lang)
    except Exception as e:
        history[-1][1] = f"{_tr(lang, 'err_generic')}：{e}"
        yield "", history, session_id, _progress_html("error", lang)


def new_chat() -> Tuple[str, List, str]:
    """Start a new chat session."""
    new_id = _new_session_id()
    return "", [], new_id


async def _check_api_status(lang: str = DEFAULT_LANG) -> str:
    """Check backend health, returning a localized status string."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(_api_url("/health"))
            if r.status_code == 200:
                return f"{_tr(lang, 'api_connected')} ({API_BASE_URL})"
    except Exception:
        pass
    return f"{_tr(lang, 'api_disconnected')} ({API_BASE_URL})"


# ── Language switch (updates every localized component) ─────────────

async def switch_language(lang: str, cur_kb: str, cur_doc_type: Optional[str]):
    """Update all UI text components when the interface language changes."""
    def t(k):
        return _tr(lang, k)

    status_val = await _check_api_status(lang)

    return (
        gr.update(value=t("title")),                                   # title_md
        gr.update(label=t("lang_label")),                              # lang_radio
        gr.update(value=t("kb_title")),                                # kb_title_md
        gr.update(label=t("kb_label"), choices=_kb_choices(lang), value=cur_kb),   # kb_kind_radio
        gr.update(value=t("settings")),                                # settings_md
        gr.update(label=t("doc_type_label"), choices=_doc_type_choices(lang), value=cur_doc_type),  # doc_type_dropdown
        gr.update(label=t("top_k_label")),                             # top_k_slider
        gr.update(value=t("session_title")),                           # session_md
        gr.update(value=t("new_chat")),                                # new_chat_btn
        gr.update(label=t("session_label")),                           # session_display
        gr.update(value=t("status_title")),                            # status_md
        gr.update(label=t("api_label"), value=status_val),             # api_status
        gr.update(label=t("chatbot_label")),                           # chatbot
        gr.update(placeholder=t("placeholder")),                       # msg_input
        gr.update(value=t("send")),                                    # send_btn
        gr.update(visible=(lang == "zh")),                             # examples_col_zh
        gr.update(visible=(lang == "en")),                             # examples_col_en
        gr.update(visible=(lang == "de")),                             # examples_col_de
        gr.update(value=""),                                           # progress_display
    )


# ── Gradio UI ───────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """Build the Gradio chat interface."""
    lang0 = DEFAULT_LANG

    with gr.Blocks(
        title=_tr(lang0, "app_title"),
        theme=gr.themes.Soft(),
        css="""
        .chatbot { height: 70vh !important; }
        .sidebar { background: var(--color-background-secondary); }
        """,
    ) as demo:
        with gr.Row():
            with gr.Column(scale=4):
                title_md = gr.Markdown(_tr(lang0, "title"))
            with gr.Column(scale=1, min_width=180):
                lang_radio = gr.Radio(
                    choices=LANGS,
                    value=lang0,
                    label=_tr(lang0, "lang_label"),
                    interactive=True,
                )

        session_id = gr.State(value=_new_session_id())

        with gr.Row():
            # Sidebar
            with gr.Column(scale=1, elem_classes="sidebar"):
                kb_title_md = gr.Markdown(_tr(lang0, "kb_title"))

                kb_kind_radio = gr.Radio(
                    choices=_kb_choices(lang0),
                    value="auto",
                    label=_tr(lang0, "kb_label"),
                    interactive=True,
                )

                settings_md = gr.Markdown(_tr(lang0, "settings"))

                doc_type_dropdown = gr.Dropdown(
                    choices=_doc_type_choices(lang0),
                    value=None,
                    label=_tr(lang0, "doc_type_label"),
                    interactive=True,
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=8,
                    step=1,
                    label=_tr(lang0, "top_k_label"),
                    interactive=True,
                )

                session_md = gr.Markdown(_tr(lang0, "session_title"))
                new_chat_btn = gr.Button(_tr(lang0, "new_chat"), variant="secondary")

                session_display = gr.Textbox(
                    value=lambda: session_id.value,
                    label=_tr(lang0, "session_label"),
                    interactive=False,
                    show_copy_button=True,
                )

                status_md = gr.Markdown(_tr(lang0, "status_title"))
                api_status = gr.Textbox(
                    value=_tr(lang0, "api_checking"),
                    label=_tr(lang0, "api_label"),
                    interactive=False,
                )

            # Chat area
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label=_tr(lang0, "chatbot_label"),
                    show_label=False,
                    height=700,
                    elem_classes="chatbot",
                    bubble_full_width=False,
                    show_copy_button=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder=_tr(lang0, "placeholder"),
                        scale=5,
                        show_label=False,
                        container=False,
                    )
                    send_btn = gr.Button(_tr(lang0, "send"), variant="primary", scale=1)

                progress_display = gr.HTML(value=_progress_html("idle", lang0))

                # Per-language example blocks (only one visible at a time).
                with gr.Column(visible=(lang0 == "zh")) as examples_col_zh:
                    gr.Examples(examples=I18N["zh"]["examples"],
                                inputs=msg_input, label=I18N["zh"]["examples_label"])
                with gr.Column(visible=(lang0 == "en")) as examples_col_en:
                    gr.Examples(examples=I18N["en"]["examples"],
                                inputs=msg_input, label=I18N["en"]["examples_label"])
                with gr.Column(visible=(lang0 == "de")) as examples_col_de:
                    gr.Examples(examples=I18N["de"]["examples"],
                                inputs=msg_input, label=I18N["de"]["examples_label"])

        # ── Event handlers ────────────────────────────────────────

        chat_inputs = [msg_input, chatbot, session_id, doc_type_dropdown,
                       top_k_slider, kb_kind_radio, lang_radio]
        chat_outputs = [msg_input, chatbot, session_id, progress_display]

        msg_input.submit(
            fn=chat_stream,
            inputs=chat_inputs,
            outputs=chat_outputs,
            api_name=None,
        )

        send_btn.click(
            fn=chat_stream,
            inputs=chat_inputs,
            outputs=chat_outputs,
            api_name=None,
        )

        new_chat_btn.click(
            fn=new_chat,
            outputs=[msg_input, chatbot, session_id],
            api_name=None,
        )

        # Language switch: update all localized components.
        lang_radio.change(
            fn=switch_language,
            inputs=[lang_radio, kb_kind_radio, doc_type_dropdown],
            outputs=[
                title_md, lang_radio, kb_title_md, kb_kind_radio, settings_md,
                doc_type_dropdown, top_k_slider, session_md, new_chat_btn,
                session_display, status_md, api_status, chatbot, msg_input,
                send_btn, examples_col_zh, examples_col_en, examples_col_de,
                progress_display,
            ],
            api_name=None,
        )

        # Check API status on load.
        demo.load(fn=_check_api_status, outputs=api_status, api_name=None)

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
