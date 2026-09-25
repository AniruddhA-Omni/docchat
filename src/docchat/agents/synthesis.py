"""Synthesis agent: picks the evidence that fits the budget and writes one grounded answer."""

from __future__ import annotations

from collections.abc import Callable

from rapidfuzz import fuzz

from docchat.agents.types import AgentContext, Evidence
from docchat.llm import LLMResult
from docchat.llm.prompts import INTENT_HINTS, SYNTHESIS_SYSTEM
from docchat.schemas import estimate_tokens

PER_SUBQUERY_FIRST = 3
AGENT_PRIORITY = {"table": 0, "code": 1, "vision": 2, "retrieval": 3}


def select_evidence(
    evidence: list[Evidence], budget_tokens: int, max_sources: int = 12
) -> list[Evidence]:
    """Coverage-aware selection: specialist results first, then the best hits of every
    sub-query, then the rest by score; near-duplicates dropped; grouped by document."""
    specialist = [e for e in evidence if e.agent != "retrieval"]
    retrieved = sorted((e for e in evidence if e.agent == "retrieval"), key=lambda e: e.score,
                       reverse=True)  # fmt: skip
    ordered: list[Evidence] = sorted(
        specialist, key=lambda e: (AGENT_PRIORITY.get(e.agent, 9), -e.score)
    )
    by_sub: dict[str, list[Evidence]] = {}
    for e in retrieved:
        by_sub.setdefault(e.sub_query, []).append(e)
    for items in by_sub.values():
        ordered.extend(items[:PER_SUBQUERY_FIRST])
    ordered.extend(retrieved)

    chosen: list[Evidence] = []
    used = 0
    for e in ordered:
        if any(e.id == c.id for c in chosen):
            continue
        if any(fuzz.ratio(e.content[:400], c.content[:400]) > 92 for c in chosen):
            continue
        cost = estimate_tokens(e.content) + 30
        if used + cost > budget_tokens and chosen:
            continue
        chosen.append(e)
        used += cost
        if len(chosen) >= max_sources:
            break
    # Group by document (documents ordered by their best source) for cross-document reasoning.
    doc_order = list(dict.fromkeys(e.doc_id for e in chosen))
    return sorted(chosen, key=lambda e: (doc_order.index(e.doc_id), chosen.index(e)))


def format_sources(sources: list[Evidence]) -> str:
    blocks = []
    for i, e in enumerate(sources, 1):
        label = e.label
        tag = {"table_result": "SQL result", "image_obs": "image observation", "code": "code"}.get(
            e.kind
        )
        if tag:
            label += f" ({tag})"
        blocks.append(f"[S{i}] {label}\n{e.content.strip()}")
    return "\n\n".join(blocks)


def build_messages(question: str, sources: list[Evidence], intent: str, history: str) -> list[dict]:
    parts = []
    if history:
        parts.append(f"Conversation so far (for context only, not a source):\n{history}\n")
    parts.append(f"Question: {question}\n")
    parts.append(f"Sources:\n{format_sources(sources)}\n")
    hint = INTENT_HINTS.get(intent)
    parts.append(f"Answer the question: {question}")
    if hint:
        parts.append(hint)
    parts.append("Remember: cite [S#] after each factual sentence, or reply NOT_FOUND.")
    return [{"role": "user", "content": "\n".join(parts)}]


def write_answer(ctx: AgentContext, messages: list[dict], think: bool,
                 on_token: Callable[[str], None] | None = None,
                 on_thinking: Callable[[str], None] | None = None) -> LLMResult:  # fmt: skip
    llm = ctx.services.chat_llm(ctx.settings)
    res = llm.chat(messages, system=SYNTHESIS_SYSTEM, tag="synthesis", think=think,
                   num_predict=3072 if think else 1024, on_token=on_token,
                   on_thinking=on_thinking)  # fmt: skip
    if res.truncated and think and not res.content.strip():
        # Thinking used the whole budget: answer again without it.
        res = llm.chat(messages, system=SYNTHESIS_SYSTEM, tag="synthesis-retry", think=False,
                       num_predict=1024, on_token=on_token)  # fmt: skip
    return res


def is_not_found(text: str) -> bool:
    t = text.strip().strip("*").strip()
    return t.upper().startswith("NOT_FOUND") or (len(t) < 40 and "NOT_FOUND" in t.upper())
