"""Conversation memory: follow-up detection, standalone-question rewriting, rolling summaries."""

from __future__ import annotations

import re

from pydantic import BaseModel

from docchat.agents.types import AgentContext
from docchat.llm import LLMError
from docchat.llm.prompts import REWRITE_SYSTEM, SUMMARY_SYSTEM

KEEP_TURNS = 6
SUMMARIZE_AFTER = 8
_FOLLOWUP_RE = re.compile(
    r"\b(it|its|they|them|their|those|these|that|this|he|she|his|her|same|above|previous|"
    r"earlier|former|latter|last one|the other)\b|^(and|also|what about|how about|why|so|then|"
    r"now|ok(ay)?|compare|same)\b",
    re.I,
)


class Rewrite(BaseModel):
    standalone: str


def is_followup(question: str, history: list[dict]) -> bool:
    if not history:
        return False
    return bool(_FOLLOWUP_RE.search(question)) or len(question.split()) <= 5


def history_text(
    history: list[dict], summary: str = "", turns: int = 3, answer_chars: int = 400
) -> str:
    lines = []
    if summary:
        lines.append(f"Earlier conversation (summary): {summary}")
    for t in history[-turns:]:
        lines.append(f"User: {t['q']}")
        lines.append(f"Assistant: {t['a'][:answer_chars]}")
    return "\n".join(lines)


def rewrite(ctx: AgentContext, question: str, history: list[dict], summary: str) -> str:
    llm = ctx.services.chat_llm(ctx.settings)
    prompt = f"Conversation:\n{history_text(history, summary)}\n\nFollow-up question: {question}"
    try:
        out = llm.json([{"role": "user", "content": prompt}], Rewrite, system=REWRITE_SYSTEM,
                       tag="rewrite", num_predict=160)  # fmt: skip
    except LLMError:
        return question
    standalone = out.standalone.strip()
    return standalone if 3 <= len(standalone) <= 600 else question


def summarize(ctx: AgentContext, history: list[dict], summary: str) -> str:
    older = history[:-KEEP_TURNS]
    if not older:
        return summary
    text = history_text(older, summary, turns=len(older), answer_chars=300)
    llm = ctx.services.chat_llm(ctx.settings)
    try:
        res = llm.chat([{"role": "user", "content": text}], system=SUMMARY_SYSTEM, tag="summary",
                       think=False, temperature=0.0, num_predict=220)  # fmt: skip
    except LLMError:
        return summary
    return res.content.strip() or summary
