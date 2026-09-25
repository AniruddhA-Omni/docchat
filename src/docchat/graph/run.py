"""Run one chat turn through the graph and stream UI-friendly events.

Events (dicts): ``status`` (progress line), ``answer_start``, ``token`` / ``thinking`` (streamed
text), ``node`` (a node finished; carries its trace entry) and finally ``done`` with the answer
payload, the full trace and the LLM call log.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from langgraph.types import Overwrite

from docchat.agents.types import AgentContext, AnswerPayload
from docchat.config import Settings
from docchat.graph.build import build_graph
from docchat.llm import CALL_LOG
from docchat.services import Services

_graph_cache: dict[int, Any] = {}


def get_graph(services: Services):
    key = id(services)
    if key not in _graph_cache:
        _graph_cache[key] = build_graph(services.checkpointer)
    return _graph_cache[key]


def stream_turn(services: Services, settings: Settings, thread_id: str, question: str,
                scope_doc_ids: list[str] | None = None) -> Iterator[dict]:  # fmt: skip
    graph = get_graph(services)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    calls: list = []
    token = CALL_LOG.set(calls)
    started = time.perf_counter()
    inputs = {"question": question, "scope_doc_ids": scope_doc_ids, "mode": settings.answer_mode}
    try:
        for part in graph.stream(inputs, config, context=AgentContext(services, settings),
                                 stream_mode=["custom", "updates"], version="v2"):  # fmt: skip
            if part["type"] == "custom":
                yield part["data"]
            elif part["type"] == "updates":
                for node_name, update in (part["data"] or {}).items():
                    entries = update.get("trace", []) if isinstance(update, dict) else []
                    if isinstance(entries, Overwrite):
                        entries = entries.value
                    for entry in entries:
                        yield {"type": "node", "node": node_name, "trace": entry}
        values = graph.get_state(config).values
    finally:
        CALL_LOG.reset(token)
    answer = AnswerPayload.model_validate(values.get("answer") or {"text": "Something went wrong."})
    yield {
        "type": "done",
        "answer": answer,
        "trace": values.get("trace", []),
        "plan": values.get("plan", {}),
        "sources": values.get("sources", []),
        "llm_calls": [c.__dict__ for c in calls],
        "seconds": round(time.perf_counter() - started, 2),
    }


def ask(services: Services, settings: Settings, thread_id: str, question: str,
        scope_doc_ids: list[str] | None = None) -> dict:  # fmt: skip
    """Blocking helper: run a turn and return the final ``done`` event."""
    final: dict = {}
    for event in stream_turn(services, settings, thread_id, question, scope_doc_ids):
        if event["type"] == "done":
            final = event
    return final


@lru_cache
def mermaid() -> str:
    """Mermaid diagram of the agent graph (shown in the UI / README)."""
    return build_graph().get_graph().draw_mermaid()
