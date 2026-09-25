"""Chat pane: transcript, streaming answer with live agent status, persistence."""

from __future__ import annotations

import streamlit as st

from docchat.config import Settings
from docchat.graph.run import stream_turn
from docchat.index.store import StoreLockedError
from docchat.llm import LLMError
from docchat.ui.answer_view import md_escape, render_answer
from docchat.ui.state import get_services

EXAMPLES = [
    "What was total revenue in FY2025, and which region grew fastest?",
    "Which regions missed their FY2025 revenue target?",
    "How many employees does Acme Robotics have?",
    "Is there a bug in the pagination of inventory_service.py?",
    "According to the chart, in which month did active robots decrease?",
    "What is the total due on invoice INV-2025-0917?",
]


def render_chat(settings: Settings) -> None:
    services = get_services()
    msgs = st.session_state.messages
    if not msgs:
        _welcome(services)
    for msg in msgs:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("payload"):
                render_answer(msg["payload"])
            else:
                st.markdown(md_escape(msg["content"]))

    question = st.chat_input("Ask about your documents…") or st.session_state.pop(
        "pending_question", None
    )
    if question:
        _answer(question, settings)


def _welcome(services) -> None:
    n_docs = len(services.catalog.ready_documents())
    st.markdown("### Ask anything about your files")
    st.caption(f"{n_docs} document(s) indexed. Answers cite file and page, tables are queried with "
               "SQL, images are read by the vision model, and every claim is verified.")  # fmt: skip
    if n_docs:
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 2].button(q, key=f"example_{i}", width="stretch"):
                st.session_state.pending_question = q
                st.rerun()


def _answer(question: str, settings: Settings) -> None:
    services = get_services()
    thread_id = st.session_state.thread_id
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md_escape(question))

    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        box = st.empty()
        streamed = ""
        final = None
        try:
            for event in stream_turn(
                services, settings, thread_id, question, st.session_state.scope
            ):
                kind = event["type"]
                if kind == "status":
                    status.update(label=event["text"])
                    status.write(event["text"])
                elif kind == "answer_start":
                    streamed = ""
                elif kind == "token":
                    streamed += event["text"]
                    box.markdown(md_escape(streamed) + " ▌")
                elif kind == "done":
                    final = event
        except (LLMError, StoreLockedError) as exc:
            status.update(label="Failed", state="error")
            box.error(f"{exc}\n\nIs Ollama running with `{settings.llm_model}` and "
                      f"`{settings.embed_model}` pulled?")  # fmt: skip
            return
        except Exception as exc:
            status.update(label="Failed", state="error")
            box.error(f"Something went wrong: {exc}")
            return
        status.update(label=f"Done in {final['seconds']:.1f}s", state="complete")
        box.empty()
        payload = {
            "question": question,
            "answer": final["answer"].model_dump(),
            "trace": final["trace"],
            "plan": final["plan"],
            "llm_calls": final["llm_calls"],
            "seconds": final["seconds"],
        }
        render_answer(payload)

    text = final["answer"].text
    st.session_state.messages.append({"role": "assistant", "content": text, "payload": payload})
    catalog = services.catalog
    catalog.touch_thread(thread_id, st.session_state.messages[0]["content"])
    catalog.add_message(thread_id, "user", question)
    catalog.add_message(thread_id, "assistant", text, payload)
    st.query_params["chat"] = thread_id
