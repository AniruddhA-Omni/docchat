"""Drive the real Streamlit app with AppTest (fake LLM, hashing embedder, no Ollama)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src" / "docchat" / "ui" / "app.py"


@pytest.fixture
def app_env(monkeypatch, tmp_path, corpus):
    for key, value in {
        "DOCCHAT_DATA_DIR": str(tmp_path / "data"),
        "DOCCHAT_MODELS_DIR": str(tmp_path / "models"),
        "DOCCHAT_OLLAMA_HOST": "http://127.0.0.1:9",  # closed port: never depends on Ollama
        "DOCCHAT_EMBED_BACKEND": "hashing",
        "DOCCHAT_RERANKER_BACKEND": "none",
        "DOCCHAT_CAPTION_IMAGES": "false",
        "DOCCHAT_ABSTAIN_DENSE_THRESHOLD": "0.1",
    }.items():
        monkeypatch.setenv(key, value)
    from docchat.config import get_settings
    from docchat.ingest.pipeline import ingest_file
    from docchat.services import Services
    from docchat.ui import state
    from tests.fakes import FakeLLM

    get_settings.cache_clear()
    st.cache_resource.clear()
    st.cache_data.clear()
    services = Services(get_settings(), llm=FakeLLM())
    for name in ("engineering_handbook.md", "support_tickets.csv"):
        ingest_file(services, get_settings(), corpus / name)
    monkeypatch.setattr(state, "build_services", lambda settings: services)
    yield services
    st.cache_resource.clear()
    services.close()
    get_settings.cache_clear()


def test_app_renders_library_and_answers(app_env):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.sidebar.error)
    assert any("Library (2)" in s.value for s in at.sidebar.subheader)

    at.chat_input[0].set_value("When are production deploys frozen?").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "December 20" in text and "[1]" in text
    assert any("confidence" in m.value.lower() for m in at.markdown)
    # The turn was persisted and can be reopened.
    assert app_env.catalog.list_threads()
    assert len(app_env.catalog.get_messages(app_env.catalog.list_threads()[0]["thread_id"])) == 2


def test_table_answer_renders(app_env):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.chat_input[0].set_value("How many support tickets are in the log?").run()
    assert not at.exception
    assert "250" in " ".join(m.value for m in at.markdown)
