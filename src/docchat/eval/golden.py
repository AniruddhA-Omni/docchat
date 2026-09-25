"""Golden Q&A set schema (written by scripts/make_demo_corpus.py, read by the evaluator)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal[
    "lookup", "table", "vision", "ocr", "code", "cross_doc", "followup", "summary", "unanswerable"
]  # fmt: skip


class ExpectedSource(BaseModel):
    file: str
    # 1-based page (PDF/DOCX) or slide (PPTX); None when any location in the file is fine.
    page: int | None = None


class GoldenItem(BaseModel):
    id: str
    question: str
    reference: str
    type: QuestionType
    answerable: bool = True
    must_cite: list[ExpectedSource] = Field(default_factory=list)
    # Canonical numbers (no thousands separators / currency) that a correct answer contains.
    expected_numbers: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    # Earlier user turns, asked before `question` in the same conversation.
    history: list[str] = Field(default_factory=list)
    requires_large: bool = False


def load_golden(path: Path) -> list[GoldenItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldenItem.model_validate_json(line) for line in lines if line.strip()]


def write_golden(path: Path, items: list[GoldenItem]) -> None:
    path.write_text("\n".join(i.model_dump_json() for i in items) + "\n", encoding="utf-8")
