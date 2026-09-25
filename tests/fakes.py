"""A scripted stand-in for the Ollama LLM so the whole graph can be tested without a model."""

from __future__ import annotations

import re
from collections.abc import Callable

from docchat.llm import CALL_LOG, CallStat, LLMResult, extract_json

Responder = Callable[[str, list[dict], str], str]


def default_responder(tag: str, messages: list[dict], system: str) -> str:
    prompt = messages[-1]["content"]
    if tag == "synthesis":
        if re.search(r"ticker|chief financial|FY2022", prompt, re.I):
            return "NOT_FOUND"
        result = re.search(r"RESULT: (\w+) = ([\d.]+)", prompt)
        if result:
            return f"The {result.group(1).replace('_', ' ')} is {result.group(2)} [S1]."
        first = re.search(r"\[S1\][^\n]*\n(.+?)(?:\n\n\[S2\]|\Z)", prompt, re.S)
        sentence = re.split(r"(?<=[.!?])\s", first.group(1).strip())[0] if first else "NOT_FOUND"
        return f"{sentence} [S1]" if first else sentence
    if tag == "text-to-sql":
        tables = re.findall(r"Table `(\w+)`", prompt)
        q = prompt.rsplit("Question:", 1)[-1].lower()
        if "europe" in q and any("orders" in t for t in tables):
            t = next(t for t in tables if "orders" in t)
            return f"```sql\nSELECT SUM(revenue) AS total_revenue FROM {t} WHERE region = 'Europe'\n```"
        if "critical" in q:
            return (
                "```sql\nSELECT COUNT(*) AS critical_tickets FROM support_tickets "
                "WHERE severity = 'critical'\n```"
            )
        if "tickets" in q:
            return "```sql\nSELECT COUNT(*) AS tickets FROM support_tickets\n```"
        return f"```sql\nSELECT COUNT(*) AS n FROM {tables[0]}\n```"
    if tag == "rewrite":
        q = prompt.rsplit("Follow-up question:", 1)[-1].strip()
        if "critical" in q:
            return '{"standalone": "How many of the support tickets are critical?"}'
        return '{"standalone": "' + q.replace('"', "") + '"}'
    if tag == "verify":
        n = len(re.findall(r"^Claim \d+:", prompt, re.M))
        return (
            '{"verdicts": ['
            + ", ".join(f'{{"i": {i}, "label": "supported"}}' for i in range(1, n + 1))
            + "]}"
        )
    if tag == "planner":
        return '{"sub_questions": [], "tools": []}'
    if tag == "summary":
        return "Earlier the user asked about the documents."
    return "ok"


class FakeLLM:
    def __init__(self, responder: Responder = default_responder, vision: bool = True) -> None:
        self.responder = responder
        self.model = "fake-llm"
        self.json_mode = "schema"
        self.supports_vision = vision
        self.supports_thinking = False
        self.capabilities = {"completion", "vision"} if vision else {"completion"}
        self.calls: list[str] = []

    def with_model(self, model: str) -> FakeLLM:
        return self

    def chat(
        self, messages, *, system=None, tag="chat", on_token=None, on_thinking=None, **_
    ) -> LLMResult:
        self.calls.append(tag)
        text = self.responder(tag, list(messages), system or "")
        log = CALL_LOG.get()
        if log is not None:
            log.append(CallStat(tag, self.model, 100, 20, 0.01))
        if on_token:
            for piece in re.findall(r"\S+\s*", text):
                on_token(piece)
        return LLMResult(content=text, model=self.model, prompt_tokens=100, output_tokens=20)

    def json(self, messages, schema, *, system=None, tag="json", **_):
        res = self.chat(messages, system=system, tag=tag)
        return schema.model_validate(extract_json(res.content))
