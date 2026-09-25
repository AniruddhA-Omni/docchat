"""Data passed between agents. Everything is plain JSON-able so LangGraph can checkpoint it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from docchat.config import Settings
from docchat.index.store import Hit
from docchat.services import Services

AgentName = Literal["retrieval", "table", "vision", "code"]
Intent = Literal["lookup", "table", "vision", "code", "compare", "summary", "chitchat"]


@dataclass
class AgentContext:
    """Injected into every graph node via LangGraph's runtime context."""

    services: Services
    settings: Settings


class Evidence(BaseModel):
    id: str
    agent: str
    kind: Literal["text", "table_result", "image_obs", "code"]
    content: str
    score: float = 0.0  # relevance / reliability estimate in [0, 1]
    doc_id: str = ""
    file_name: str = ""
    file_type: str = ""
    location: str = ""  # "p.4", "slide 3", "L10-42", "sheet Orders"
    page: int | None = None
    page_end: int | None = None
    page_approx: bool = False
    bboxes: list[list[float]] = Field(default_factory=list)
    line_start: int | None = None
    line_end: int | None = None
    section: str = ""
    sheet: str | None = None
    table_name: str | None = None
    sql: str | None = None
    row_ids: list[int] = Field(default_factory=list)
    image_path: str | None = None
    chunk_kind: str = ""
    ocr_conf: float | None = None
    sub_query: str = ""

    @property
    def label(self) -> str:
        parts = [self.file_name, self.location, self.section]
        return " · ".join(p for p in parts if p)


def evidence_from_hit(hit: Hit, agent: str = "retrieval", sub_query: str = "") -> Evidence:
    c = hit.chunk
    return Evidence(
        id=c.chunk_id,
        agent=agent,
        kind="code" if c.kind == "code" else "text",
        content=c.text,
        score=round(hit.relevance, 4),
        doc_id=c.doc_id,
        file_name=c.file_name,
        file_type=c.file_type,
        location=c.location(),
        page=c.page_start,
        page_end=c.page_end,
        page_approx=c.page_approx,
        bboxes=c.bboxes,
        line_start=c.line_start,
        line_end=c.line_end,
        section=" > ".join(c.section_path[-2:]),
        sheet=c.sheet,
        table_name=c.table_name,
        image_path=c.image_path,
        chunk_kind=str(c.kind),
        ocr_conf=c.ocr_conf,
        sub_query=sub_query,
    )


class SubQuery(BaseModel):
    query: str
    doc_ids: list[str] | None = None
    label: str = ""


class Plan(BaseModel):
    intent: Intent = "lookup"
    agents: list[AgentName] = Field(default_factory=lambda: ["retrieval"])
    sub_queries: list[SubQuery] = Field(default_factory=list)
    table_names: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)  # chunk ids of images to inspect
    symbols: list[str] = Field(default_factory=list)
    doc_filter: list[str] | None = None  # docs explicitly named in the question
    reasons: list[str] = Field(default_factory=list)
    used_llm: bool = False


class Citation(BaseModel):
    n: int
    evidence_id: str
    agent: str
    file_name: str
    doc_id: str
    file_type: str = ""
    location: str = ""
    page: int | None = None
    page_approx: bool = False
    bboxes: list[list[float]] = Field(default_factory=list)
    line_start: int | None = None
    line_end: int | None = None
    sheet: str | None = None
    table_name: str | None = None
    sql: str | None = None
    row_ids: list[int] = Field(default_factory=list)
    image_path: str | None = None
    snippet: str = ""
    quote: str = ""  # the sentence in the source that best supports the citing claim
    score: float = 0.0


class Verdict(BaseModel):
    sentence: str
    citations: list[int] = Field(default_factory=list)
    label: Literal["supported", "partial", "unsupported", "uncited", "not_factual"] = "supported"
    reason: str = ""
    checked_by: str = "rules"  # rules | llm


class Confidence(BaseModel):
    score: float
    band: Literal["high", "medium", "low"]
    verified: float
    retrieval: float
    coverage: float
    numeric: float
    notes: list[str] = Field(default_factory=list)


class AnswerPayload(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    confidence: Confidence | None = None
    abstained: bool = False
    abstain_reason: str = ""
    nearest_sources: list[str] = Field(default_factory=list)
    standalone_question: str = ""
    removed_sentences: list[str] = Field(default_factory=list)
    thinking: str = ""
