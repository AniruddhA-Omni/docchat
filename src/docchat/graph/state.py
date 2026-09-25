"""LangGraph state. Keys with reducers are written by parallel agents; per-turn keys are reset
with ``Overwrite`` at the start of every turn, while ``history`` / ``summary`` persist in the
thread's checkpoint (conversation memory)."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_evidence(current: list[dict], new: list[dict]) -> list[dict]:
    """Union by evidence id, keeping the higher score."""
    by_id = {e["id"]: e for e in current}
    for e in new:
        old = by_id.get(e["id"])
        if old is None or e.get("score", 0) > old.get("score", 0):
            by_id[e["id"]] = e
    return list(by_id.values())


class GraphState(TypedDict, total=False):
    # --- persisted across turns ---------------------------------------------------------------
    history: Annotated[list[dict], operator.add]  # {q, standalone, a, docs}
    summary: str
    active_doc_ids: list[str]
    last_sql: dict | None
    # --- turn input -----------------------------------------------------------------------------
    question: str
    scope_doc_ids: list[str] | None  # None = all ready documents
    mode: str  # fast | balanced | deep
    # --- per turn -------------------------------------------------------------------------------
    standalone: str
    followup: bool
    doc_ids: list[str]
    plan: dict[str, Any]
    evidence: Annotated[list[dict], merge_evidence]
    trace: Annotated[list[dict], operator.add]
    sources: list[dict]  # evidence selected for synthesis, in [S#] order
    draft: str
    thinking: str
    abstain: str  # reason, empty if answering
    answer: dict[str, Any]
