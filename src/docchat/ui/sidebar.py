"""Sidebar: models and modes, parsing backends, uploads, document scope, chat history."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from docchat.config import Settings
from docchat.extras import OCR_BACKENDS, PDF_PARSERS, backend_available, install_hint
from docchat.ingest.detect import SUPPORTED_EXTENSIONS, detect_format
from docchat.ingest.pipeline import delete_document
from docchat.ingest.safety import safe_filename
from docchat.ui.state import (
    current_settings,
    get_jobs,
    get_services,
    new_chat,
    ollama_status,
    open_chat,
    set_override,
)

_EMBEDDING_HINTS = ("bge", "embed", "minilm", "e5-", "gte-", "snowflake")
_MODES = {"fast": "⚡ Fast", "balanced": "⚖️ Balanced", "deep": "🧠 Deep"}
_STATUS_ICON = {"ready": "✅", "processing": "⏳", "error": "❌"}
DEMO_DIR = Path("demo_corpus")


def render_sidebar() -> Settings:
    settings = current_settings()
    with st.sidebar:
        st.title("📄 DocChat")
        st.caption("Local multi-agent chat over your files · runs fully offline")
        _model_controls(settings)
        _mode_controls(settings)
        _parsing_controls(settings)
        st.divider()
        _upload_controls(current_settings())
        _ingest_progress()
        _library_scope()
        st.divider()
        _chat_history()
    return current_settings()


def _model_controls(settings: Settings) -> None:
    status = ollama_status(settings.ollama_host)
    if not status.reachable:
        st.error(f"Ollama is not reachable at `{settings.ollama_host}`. "
                 "Start the Ollama app (or `ollama serve`) and refresh.")  # fmt: skip
        return
    chat_models = [m for m in status.models if not m.lower().startswith(_EMBEDDING_HINTS)]
    if settings.llm_model not in chat_models:
        chat_models.insert(0, settings.llm_model)
    choice = st.selectbox(
        "Chat & vision model", chat_models, index=chat_models.index(settings.llm_model),
        key="llm_model_select",
        help="qwen3.5 and gemma4 are both multimodal, so one model serves every agent. "
             "Change the default with DOCCHAT_LLM_MODEL in .env.",
    )  # fmt: skip
    set_override("llm_model", choice)
    for model in dict.fromkeys([choice, settings.embed_model]):
        if not status.has(model):
            st.warning(f"`{model}` is not pulled. Run `ollama pull {model}`")


def _mode_controls(settings: Settings) -> None:
    mode = st.segmented_control(
        "Answer mode", list(_MODES), format_func=_MODES.get, default=settings.answer_mode,
        key="answer_mode_select",
        help="Fast: rule-based verification only. Balanced: + LLM claim check. "
             "Deep: + thinking mode and one revision pass for unverified claims.",
    )  # fmt: skip
    set_override("answer_mode", mode)


def _parsing_controls(settings: Settings) -> None:
    with st.expander("Parsing & OCR backends"):
        pdf = _backend_select("PDF parser", "pdf", PDF_PARSERS, settings.pdf_parser)
        set_override("pdf_parser", pdf)
        ocr = _backend_select("OCR engine", "ocr", OCR_BACKENDS, settings.ocr_backend)
        set_override("ocr_backend", ocr)
        caption = st.toggle("Describe images with the vision model at upload",
                            value=settings.caption_images, key="caption_toggle")  # fmt: skip
        set_override("caption_images", caption)
        st.caption("Changing the parser re-indexes PDFs the next time they are uploaded.")


def _backend_select(label: str, kind: str, table: dict, current: str) -> str:
    names = list(table)
    available = {n: backend_available(kind, n) for n in names}
    choice = st.selectbox(label, names, index=names.index(current), key=f"{kind}_backend_select",
                          format_func=lambda n: n if available[n] else f"{n} (not installed)")  # fmt: skip
    if not available[choice]:
        st.caption(f"Install with `{install_hint(kind, choice)}` — falling back to the default.")
    return choice


def _upload_controls(settings: Settings) -> None:
    files = st.file_uploader("Add files", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True,
                             key=f"uploader_{st.session_state.uploader_key}")  # fmt: skip
    cols = st.columns(2)
    if files and cols[0].button(f"Index {len(files)} file(s)", type="primary", width="stretch"):
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in files:
            dest = settings.uploads_dir / safe_filename(f.name)
            dest.write_bytes(f.getbuffer())
            paths.append(dest)
        get_jobs().submit(paths, settings)
        st.session_state.uploader_key += 1  # clears the uploader
        st.rerun()
    if DEMO_DIR.is_dir() and cols[1 if files else 0].button("Load demo corpus", width="stretch",
                                                            help="Index every file in ./demo_corpus"):  # fmt: skip
        paths = sorted(
            p for p in DEMO_DIR.iterdir() if detect_format(p) and p.name != "manifest.json"
        )
        get_jobs().submit(paths, settings)
        st.rerun()


@st.fragment(run_every=1.0)
def _ingest_progress() -> None:
    jobs = get_jobs()
    active = jobs.active()
    recent = jobs.recent()
    if not recent:
        return
    for j in recent:
        icon = {"queued": "🕒", "running": "⏳", "ready": "✅", "skipped": "♻️", "error": "❌"}[
            j.status
        ]
        st.caption(f"{icon} **{j.file_name}** — {j.message}")
    if not active and st.session_state.get("_jobs_were_active"):
        st.session_state["_jobs_were_active"] = False
        st.rerun(scope="app")  # refresh the library once everything is indexed
    elif active:
        st.session_state["_jobs_were_active"] = True


def _library_scope() -> None:
    services = get_services()
    docs = services.catalog.list_documents()
    ready = [d for d in docs if d.status == "ready"]
    st.subheader(f"Library ({len(ready)})")
    if not docs:
        st.caption("No documents yet — upload files or load the demo corpus.")
        return
    use_all = st.toggle(
        "Answer from all documents", value=st.session_state.scope is None, key="scope_all"
    )
    selected = set(st.session_state.scope or [])
    for d in docs:
        icon = _STATUS_ICON.get(d.status, "•")
        label = f"{icon} {d.file_name}"
        c1, c2 = st.columns([0.85, 0.15])
        if use_all or d.status != "ready":
            c1.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
            if d.status == "error":
                c1.caption(f"⚠️ {d.error}")
        else:
            if c1.checkbox(label, value=d.doc_id in selected, key=f"scope_{d.doc_id}"):
                selected.add(d.doc_id)
            else:
                selected.discard(d.doc_id)
        if c2.button("🗑", key=f"del_{d.doc_id}", help=f"Remove {d.file_name}"):
            delete_document(services, current_settings(), d.doc_id)
            st.rerun()
    st.session_state.scope = None if use_all else sorted(selected)


def _chat_history() -> None:
    st.button("➕ New chat", on_click=new_chat, width="stretch")
    threads = get_services().catalog.list_threads(limit=10)
    if threads:
        st.caption("Recent chats")
    for t in threads:
        current = t["thread_id"] == st.session_state.thread_id
        st.button(("▶ " if current else "") + (t["title"] or "Untitled"), key=f"thread_{t['thread_id']}",
                  on_click=open_chat, args=(t["thread_id"],), width="stretch", type="tertiary")  # fmt: skip


def library_file_path(doc_id: str) -> Path | None:
    rec = get_services().catalog.get_document(doc_id)
    return Path(rec.path) if rec else None
