"""Structure-aware chunking.

Rules: chunks never cross a heading boundary; tables, figures, slide notes and code symbols are
their own chunks (oversized tables are split by row groups that repeat the header); prose is packed
up to ``chunk_target_tokens`` and split on sentences when a single block is too long. Every chunk
keeps file name, page range, bounding boxes / line range and the heading breadcrumb, which is also
prepended as ``ctx_header`` when embedding and reranking.
"""

from __future__ import annotations

import re

from docchat.config import Settings
from docchat.ingest.tables import rows_markdown
from docchat.schemas import Chunk, Element, Kind, ParsedDocument, estimate_tokens

CHUNKER_VERSION = "1"
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_PROSE = {Kind.PARAGRAPH, Kind.LIST, Kind.OCR_TEXT, Kind.CAPTION}
_STANDALONE = {Kind.TABLE, Kind.TABLE_CARD, Kind.FIGURE, Kind.NOTES, Kind.CODE}
MIN_PAGE_FLUSH_TOKENS = 120


def assign_sections(elements: list[Element]) -> None:
    """Fill ``section_path`` from the heading hierarchy (in reading order)."""
    stack: list[tuple[int, str]] = []
    for el in elements:
        if el.kind == Kind.HEADING:
            level = el.heading_level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, el.text.strip()[:120]))
            el.section_path = [t for _, t in stack]
        else:
            el.section_path = [t for _, t in stack]


def split_sentences(text: str, max_tokens: int, overlap: int = 1) -> list[str]:
    """Split long prose into <= max_tokens pieces on sentence boundaries (1-sentence overlap)."""
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return split_lines(text, max_tokens)
    pieces, current = [], []
    for sent in sentences:
        if current and estimate_tokens(" ".join([*current, sent])) > max_tokens:
            pieces.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            if estimate_tokens(" ".join([*current, sent])) > max_tokens:
                current = []
        if estimate_tokens(sent) > max_tokens:
            pieces.extend(split_lines(sent, max_tokens))
            continue
        current.append(sent)
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_lines(text: str, max_tokens: int) -> list[str]:
    """Split on newlines (then hard-wrap) so that each piece fits the token budget."""
    max_chars = int(max_tokens * 3.2)
    pieces, current = [], ""
    for line in text.split("\n"):
        while len(line) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            cut = line.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append(line[:cut])
            line = line[cut:].lstrip()
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = line
        else:
            current = candidate
    if current.strip():
        pieces.append(current)
    return pieces


class _Builder:
    def __init__(self, parsed: ParsedDocument, settings: Settings) -> None:
        self.parsed = parsed
        self.target = settings.chunk_target_tokens
        self.max = settings.chunk_max_tokens
        self.chunks: list[Chunk] = []
        self.buffer: list[Element] = []

    # -- emit ------------------------------------------------------------------------------------
    def emit(self, elements: list[Element], text: str, kind: Kind, **extra) -> None:
        pages = [e.page for e in elements if e.page]
        lines_start = [e.line_start for e in elements if e.line_start]
        lines_end = [e.line_end for e in elements if e.line_end]
        section = elements[0].section_path
        confs = [e.meta["ocr_conf"] for e in elements if "ocr_conf" in e.meta]
        chunk = Chunk(
            chunk_id=f"{self.parsed.doc_id}:{len(self.chunks)}",
            doc_id=self.parsed.doc_id,
            ordinal=len(self.chunks),
            file_name=self.parsed.file_name,
            file_type=self.parsed.format,
            kind=kind,
            text=text.strip(),
            ctx_header="",
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            bboxes=[[e.page or 1, *e.bbox] for e in elements if e.bbox],
            line_start=min(lines_start) if lines_start else None,
            line_end=max(lines_end) if lines_end else None,
            section_path=section,
            parser=self.parsed.parser,
            ocr_engine=self.parsed.ocr_engine if confs or kind == Kind.FIGURE else None,
            ocr_conf=round(sum(confs) / len(confs), 3) if confs else None,
            page_approx=any(e.meta.get("page_approx") for e in elements),
            language=self.parsed.language,
        )
        for key, value in extra.items():
            setattr(chunk, key, value)
        chunk.token_count = estimate_tokens(chunk.embed_text)
        chunk.ctx_header = _ctx_header(chunk)
        self.chunks.append(chunk)

    def flush(self) -> None:
        if not self.buffer:
            return
        els, self.buffer = self.buffer, []
        text = "\n\n".join(e.text for e in els)
        if estimate_tokens(text) <= self.max:
            self.emit(els, text, _dominant_kind(els))
            return
        for piece in split_sentences(text, self.target):
            # Attribute each piece to the elements whose text it overlaps (page / bbox precision).
            owners = [e for e in els if _overlaps(piece, e.text)] or els
            self.emit(owners, piece, _dominant_kind(owners))

    # -- elements --------------------------------------------------------------------------------
    def add(self, el: Element) -> None:
        if el.kind in (Kind.HEADER, Kind.FOOTER):
            return
        if el.kind == Kind.HEADING:
            self.flush()
            return
        if el.kind in _STANDALONE:
            self.flush()
            self.standalone(el)
            return
        if self.buffer:
            prev = self.buffer[-1]
            buffered = estimate_tokens("\n\n".join(e.text for e in self.buffer))
            new_page = el.page and prev.page and el.page != prev.page
            if (
                el.section_path != prev.section_path
                or buffered + estimate_tokens(el.text) > self.target
                or (new_page and buffered >= MIN_PAGE_FLUSH_TOKENS)
                or (self.parsed.format == "pptx" and new_page)
            ):
                self.flush()
        self.buffer.append(el)

    def standalone(self, el: Element) -> None:
        meta = el.meta
        extra = {
            "image_path": el.image_path,
            "table_name": meta.get("table_name"),
            "sheet": meta.get("sheet"),
            "symbol": meta.get("symbol"),
            "language": meta.get("language") or self.parsed.language,
        }
        limit = self.max + 200 if el.kind == Kind.TABLE_CARD else self.max
        if estimate_tokens(el.text) <= limit:
            parts = [el.text]
        elif el.kind == Kind.TABLE and el.table_rows:
            parts = _split_table(el, self.max)
        elif el.kind == Kind.CODE:
            parts = _split_code(el.text, self.max)
        else:
            parts = split_sentences(el.text, self.target)
        for part in parts:
            self.emit([el], part, el.kind, **extra)


def _dominant_kind(els: list[Element]) -> Kind:
    kinds = [e.kind for e in els]
    if Kind.OCR_TEXT in kinds:
        return Kind.OCR_TEXT
    return Kind.LIST if kinds.count(Kind.LIST) > len(kinds) / 2 else Kind.PARAGRAPH


def _overlaps(piece: str, text: str) -> bool:
    probe = text.strip()[:60]
    return bool(probe) and (probe in piece or piece[:60] in text)


def _split_table(el: Element, max_tokens: int) -> list[str]:
    rows = el.table_rows or []
    header, body = rows[0], rows[1:]
    caption = el.meta.get("caption", "")
    parts, group = [], []
    for row in body:
        candidate = rows_markdown([header, *group, row])
        if group and estimate_tokens(caption + candidate) > max_tokens:
            parts.append(rows_markdown([header, *group]))
            group = []
        group.append(row)
    if group:
        parts.append(rows_markdown([header, *group]))
    n = len(parts)
    return [f"{caption}\n(part {i}/{n})\n{p}".strip() for i, p in enumerate(parts, 1)]


def _split_code(text: str, max_tokens: int) -> list[str]:
    fence = text.split("\n", 1)[0] if text.startswith("```") else "```"
    body = text.removeprefix(fence).removesuffix("```").strip("\n")
    return [f"{fence}\n{p}\n```" for p in split_lines(body, max_tokens - 10)]


def _ctx_header(chunk: Chunk) -> str:
    parts = [chunk.file_name]
    if chunk.section_path:
        parts.append(" > ".join(chunk.section_path[-3:]))
    if chunk.sheet:
        parts.append(f"sheet {chunk.sheet}")
    if chunk.symbol:
        parts.append(f"symbol {chunk.symbol}")
    loc = chunk.location()
    if loc:
        parts.append(loc)
    return " | ".join(parts)


def chunk_document(parsed: ParsedDocument, settings: Settings) -> list[Chunk]:
    assign_sections(parsed.elements)
    builder = _Builder(parsed, settings)
    for el in parsed.elements:
        builder.add(el)
    builder.flush()
    return builder.chunks
