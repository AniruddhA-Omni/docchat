"""Streamlit entry point: ``uv run docchat`` or ``uv run streamlit run src/docchat/ui/app.py``."""

from __future__ import annotations

import streamlit as st

from docchat.config import apply_runtime_env, get_settings
from docchat.ui.about import render_about
from docchat.ui.chat import render_chat
from docchat.ui.library import render_library
from docchat.ui.sidebar import render_sidebar
from docchat.ui.state import init_state

st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
apply_runtime_env(get_settings())
init_state()

settings = render_sidebar()
chat_tab, library_tab, about_tab = st.tabs(["💬 Chat", "📚 Library", "⚙️ How it works"])
with chat_tab:
    render_chat(settings)
with library_tab:
    render_library()
with about_tab:
    render_about(settings)
