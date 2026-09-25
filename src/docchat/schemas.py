"""Core data model shared by parsers, the chunker, the index and the agents."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1; top-left origin (PDF points / pixels)


class Kind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    TABLE_CARD = "table_card"  # schema + profile summary of a spreadsheet / extracted table
    FIGURE = "figure"
    CAPTION = "caption"
    CODE = "code"
    OCR_TEXT = "ocr_text"
    NOTES = "notes"  # slide speaker notes
    HEADER = "header"  # page header/footer: parsed, never indexed
    FOOTER = "footer"


class Element(BaseModel):
    """One structural unit emitted by a parser, in reading order."""

    kind: Kind
    text: str
    page: int | None = None  # 1-based page (PDF), approximate page (DOCX) or slide (PPTX)
    bbox: BBox | None = None
    heading_level: int | None = None
    section_path: list[str] = Field(default_factory=list)  # filled by chunking.assign_sections
    line_start: int | None = None
    line_end: int | None = None
    table_rows: list[list[str]] | None = None  # header row first, for document tables
    image_path: str | None = None  # crop / original image for the vision agent
    meta: dict[str, Any] = Field(default_factory=dict)


@dataclass
class TableData:
    """A structured table that the Table agent can query with SQL."""

    df: pd.DataFrame
    columns: dict[str, str]  # sql column name -> original header text
    label: str  # human name, e.g. "sales_2025.xlsx / Orders"
    sheet: str | None = None
    title: str | None = None
    page: int | None = None
    header_row: int | None = None  # 1-based source row of the (last) header line
    notes: list[str] = field(default_factory=list)
    name: str = ""  # SQL identifier, assigned when registered in the catalog


@dataclass
class ParsedDocument:
    doc_id: str
    file_name: str
    path: Path
    format: str
    parser: str
    elements: list[Element] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    page_count: int | None = None
    ocr_engine: str | None = None
    language: str | None = None  # code files
    symbols: list[dict[str, Any]] = field(default_factory=list)  # code: {symbol, kind, lines}
    warnings: list[str] = field(default_factory=list)


class Chunk(BaseModel):
    """Retrieval unit. Stored as the Qdrant payload; ``text`` is what gets cited."""

    chunk_id: str
    doc_id: str
    ordinal: int
    file_name: str
    file_type: str
    kind: Kind
    text: str
    ctx_header: str
    page_start: int | None = None
    page_end: int | None = None
    bboxes: list[list[float]] = Field(default_factory=list)  # [page, x0, y0, x1, y1]
    line_start: int | None = None
    line_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    sheet: str | None = None
    table_name: str | None = None
    symbol: str | None = None
    language: str | None = None
    image_path: str | None = None
    parser: str = ""
    ocr_engine: str | None = None
    ocr_conf: float | None = None
    page_approx: bool = False
    token_count: int = 0

    @property
    def embed_text(self) -> str:
        """Text used for dense/sparse embedding and reranking (context header + body)."""
        return f"{self.ctx_header}\n{self.text}"

    def location(self) -> str:
        """Short human-readable location for citations."""
        if self.page_start:
            prefix = "≈p." if self.page_approx else "p."
            if self.file_type == "pptx":
                prefix = "slide "
            pages = (
                f"{self.page_start}"
                if self.page_start == self.page_end or not self.page_end
                else f"{self.page_start}-{self.page_end}"
            )
            return f"{prefix}{pages}"
        if self.line_start:
            return f"L{self.line_start}-{self.line_end}"
        if self.sheet:
            return f"sheet {self.sheet}"
        return ""


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for bge-m3 / bge-reranker (XLM-R style tokenizers)."""
    return max(1, math.ceil(len(text) / 3.2))


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
