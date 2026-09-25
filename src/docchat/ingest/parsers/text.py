"""Plain-text and Markdown parsers. Both keep 1-based source line numbers for citations."""

from __future__ import annotations

import re
from pathlib import Path

from charset_normalizer import from_bytes
from markdown_it import MarkdownIt

from docchat.ingest.context import ParseContext
from docchat.ingest.tables import rows_markdown, rows_to_table
from docchat.schemas import Element, Kind, ParsedDocument

_BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+")
_UNDERLINE = re.compile(r"^\s*(=+|-+)\s*$")


def read_text(path: Path) -> str:
    """Decode a text file, detecting its encoding (UTF-8 BOM, cp1252, UTF-16...)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig",):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    best = from_bytes(raw).best()
    return str(best) if best is not None else raw.decode("latin-1")


def parse_text(path: Path, ctx: ParseContext) -> ParsedDocument:
    text = read_text(path).replace("\r\n", "\n")
    parsed = ParsedDocument(doc_id=ctx.doc_id, file_name=path.name, path=path, format="text",
                            parser="text")  # fmt: skip
    lines = text.split("\n")
    block: list[str] = []
    start = 1

    def flush(end: int) -> None:
        if not block:
            return
        first = block[0].strip()
        # "Decisions" followed by a bullet list -> heading + list.
        if len(block) > 1 and _looks_like_heading(first) and _BULLET.match(block[1]):
            parsed.elements.append(Element(kind=Kind.HEADING, text=first, heading_level=2,
                                           line_start=start, line_end=start))  # fmt: skip
            parsed.elements.append(Element(kind=Kind.LIST, text="\n".join(block[1:]).strip(),
                                           line_start=start + 1, line_end=end))  # fmt: skip
            return
        body = "\n".join(block).strip()
        next_is_rule = len(block) >= 2 and _UNDERLINE.match(block[1])
        if next_is_rule or (len(block) == 1 and _looks_like_heading(first)):
            kind, level = Kind.HEADING, 1 if next_is_rule and "=" in block[1] else 2
            body = first
        else:
            kind = Kind.LIST if all(_BULLET.match(b) or b.startswith(" ") for b in block[1:]) \
                and _BULLET.match(block[min(1, len(block) - 1)]) else Kind.PARAGRAPH  # fmt: skip
            level = None
        parsed.elements.append(Element(kind=kind, text=body, heading_level=level,
                                       line_start=start, line_end=end))  # fmt: skip
        if kind == Kind.HEADING and len(block) > 2:  # heading + rule + more text
            rest = "\n".join(block[2:]).strip()
            parsed.elements.append(Element(kind=Kind.PARAGRAPH, text=rest,
                                           line_start=start + 2, line_end=end))  # fmt: skip

    for i, line in enumerate(lines, 1):
        if line.strip():
            if not block:
                start = i
            block.append(line.rstrip())
        else:
            flush(i - 1)
            block = []
    flush(len(lines))
    parsed.page_count = None
    return parsed


def _looks_like_heading(line: str) -> bool:
    words = line.split()
    return (
        0 < len(words) <= 8
        and not line.endswith((".", ",", ";", ":"))
        and (line.isupper() or line.istitle() or line[0].isupper())
        and not _BULLET.match(line)
        and len(line) <= 60
    )


# ------------------------------------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------------------------------------
def parse_markdown(path: Path, ctx: ParseContext) -> ParsedDocument:
    text = read_text(path).replace("\r\n", "\n")
    parsed = ParsedDocument(doc_id=ctx.doc_id, file_name=path.name, path=path, format="markdown",
                            parser="markdown-it")  # fmt: skip
    markdown_into(parsed, text)
    return parsed


def markdown_into(parsed: ParsedDocument, text: str, page: int | None = None) -> None:
    """Append Markdown blocks (with source line numbers, or a fixed page) to ``parsed``."""
    path = parsed.path
    src_lines = text.split("\n")
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(text)
    first_new, first_table = len(parsed.elements), len(parsed.tables)
    i, n_tables = 0, len(parsed.tables)
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            inline = tokens[i + 1]
            start, end = _lines(tok)
            parsed.elements.append(Element(kind=Kind.HEADING, text=inline.content.strip(),
                                           heading_level=int(tok.tag[1]),
                                           line_start=start, line_end=end))  # fmt: skip
            i += 3
            continue
        if (
            tok.type in ("bullet_list_open", "ordered_list_open", "blockquote_open")
            and tok.level == 0
        ):
            close = tok.type.replace("_open", "_close")
            j = i + 1
            while j < len(tokens) and not (tokens[j].type == close and tokens[j].level == 0):
                j += 1
            start, end = _lines(tok)
            body = "\n".join(src_lines[start - 1 : end]).strip()
            kind = Kind.PARAGRAPH if tok.type == "blockquote_open" else Kind.LIST
            parsed.elements.append(Element(kind=kind, text=body, line_start=start, line_end=end))
            i = j + 1
            continue
        if tok.type == "paragraph_open" and tok.level == 0:
            start, end = _lines(tok)
            parsed.elements.append(Element(kind=Kind.PARAGRAPH, text=tokens[i + 1].content.strip(),
                                           line_start=start, line_end=end))  # fmt: skip
            i += 3
            continue
        if tok.type in ("fence", "code_block"):
            start, end = _lines(tok)
            lang = (tok.info or "").strip().split(" ")[0]
            body = f"```{lang}\n{tok.content.rstrip()}\n```"
            parsed.elements.append(Element(kind=Kind.CODE, text=body, line_start=start,
                                           line_end=end, meta={"language": lang}))  # fmt: skip
            i += 1
            continue
        if tok.type == "table_open":
            j, rows, row = i + 1, [], []
            while tokens[j].type != "table_close":
                t = tokens[j]
                if t.type == "tr_open":
                    row = []
                elif t.type == "inline":
                    row.append(t.content.strip())
                elif t.type == "tr_close":
                    rows.append(row)
                j += 1
            start, end = _lines(tok)
            if len(rows) >= 2:
                n_tables += 1
                parsed.elements.append(Element(kind=Kind.TABLE, text=rows_markdown(rows),
                                               table_rows=rows, line_start=start, line_end=end,
                                               meta={"table_idx": len(parsed.tables)}))  # fmt: skip
                table = rows_to_table(rows, label=f"{path.name} table {n_tables}")
                table.title = f"Table {n_tables} in {path.name} (lines {start}-{end})"
                parsed.tables.append(table)
            i = j + 1
            continue
        if tok.type == "html_block":
            start, end = _lines(tok)
            parsed.elements.append(Element(kind=Kind.PARAGRAPH, text=tok.content.strip(),
                                           line_start=start, line_end=end))  # fmt: skip
        i += 1
    if page is not None:
        for el in parsed.elements[first_new:]:
            el.page, el.line_start, el.line_end = page, None, None
        for t in parsed.tables[first_table:]:
            t.page = page


def _lines(tok) -> tuple[int, int]:
    if tok.map:
        return tok.map[0] + 1, max(tok.map[0] + 1, tok.map[1])
    return 0, 0
