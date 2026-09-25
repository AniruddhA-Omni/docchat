"""Streamlit session state and cached, process-wide resources."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import streamlit as st

from docchat.config import Settings, get_settings
from docchat.llm import OllamaStatus, check_ollama
from docchat.services import Services
from docchat.ui.jobs import IngestJobs


def build_services(settings: Settings) -> Services:
    """Factory (monkeypatched in UI tests to inject a fake LLM)."""
    return Services(settings)


@st.cache_resource(show_spinner="Starting DocChat…")
def get_services() -> Services:
    return build_services(get_settings())


@st.cache_resource
def get_jobs() -> IngestJobs:
    return IngestJobs(get_services())


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])  # [{role, content, payload?}]
    ss.setdefault("thread_id", st.query_params.get("chat") or uuid4().hex)
    ss.setdefault("overrides", {})  # per-session Settings overrides chosen in the sidebar
    ss.setdefault("uploader_key", 0)
    ss.setdefault("scope", None)  # None = all documents; else list of doc_ids
    ss.setdefault("pending_question", None)
    if not ss.messages and st.query_params.get("chat"):
        ss.messages = get_services().catalog.get_messages(ss.thread_id)


def current_settings() -> Settings:
    return get_settings().model_copy(update=st.session_state.overrides)


def set_override(key: str, value: Any) -> None:
    if value is None or getattr(get_settings(), key) == value:
        st.session_state.overrides.pop(key, None)
    else:
        st.session_state.overrides[key] = value


def new_chat() -> None:
    st.session_state.messages = []
    st.session_state.thread_id = uuid4().hex
    st.query_params.pop("chat", None)


def open_chat(thread_id: str) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.messages = get_services().catalog.get_messages(thread_id)
    st.query_params["chat"] = thread_id


@st.cache_data(ttl=15, show_spinner=False)
def ollama_status(host: str) -> OllamaStatus:
    return check_ollama(get_settings().model_copy(update={"ollama_host": host}))
