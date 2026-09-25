"""Unstructured PDF backend (extra ``unstructured``). ``hi_res`` needs poppler + tesseract on
PATH; without them the ``fast`` strategy (pdfminer text + layout heuristics) is used."""

from __future__ import annotations

import shutil
from pathlib import Path

from docchat.ingest.context import ParseContext
from docchat.ingest.tables import rows_markdown, rows_to_table
from docchat.schemas import Element, Kind, ParsedDocument

_KINDS = {
    "Title": Kind.HEADING, "NarrativeText": Kind.PARAGRAPH, "Text": Kind.PARAGRAPH,
    "ListItem": Kind.LIST, "Table": Kind.TABLE, "Image": Kind.FIGURE, "FigureCaption": Kind.CAPTION,
    "Header": Kind.HEADER, "Footer": Kind.FOOTER, "PageNumber": Kind.FOOTER, "Formula": Kind.PARAGRAPH,
    "Address": Kind.PARAGRAPH, "EmailAddress": Kind.PARAGRAPH, "CodeSnippet": Kind.CODE,
}  # fmt: skip


def _html_rows(html: str) -> list[list[str]]:
    from io import StringIO

    import pandas as pd

    try:
        df = pd.read_html(StringIO(html))[0]
    except (ValueError, ImportError):
        return []
    return [
        [str(c) for c in df.columns],
        *[[str(v) for v in r] for r in df.itertuples(index=False)],
    ]


def _page_sizes(path: Path) -> dict[int, tuple[float, float]]:
    """Page sizes in PDF points (to map Unstructured coordinates onto the page for highlights)."""
    try:
        import pymupdf
    except (ImportError, OSError):
        return {}
    with pymupdf.open(path) as doc:
        return {p.number + 1: (p.rect.width, p.rect.height) for p in doc}


def parse_pdf(path: Path, ctx: ParseContext) -> ParsedDocument:
    from unstructured.partition.pdf import partition_pdf

    hi_res = shutil.which("pdftoppm") and shutil.which("tesseract")
    strategy = "hi_res" if hi_res else "fast"
    ctx.report(f"{path.name}: running Unstructured ({strategy})")
    items = partition_pdf(filename=str(path), strategy=strategy, infer_table_structure=bool(hi_res))
    parsed = ParsedDocument(doc_id=ctx.doc_id, file_name=path.name, path=path, format="pdf",
                            parser=f"unstructured-{strategy}")  # fmt: skip
    page_sizes = _page_sizes(path)
    for it in items:
        kind = _KINDS.get(it.category, Kind.PARAGRAPH)
        md = it.metadata
        page = getattr(md, "page_number", None)
        bbox = None
        coords = getattr(md, "coordinates", None)
        size = page_sizes.get(page) if page else None
        if coords is not None and coords.points and size and getattr(coords.system, "width", None):
            xs = [p[0] for p in coords.points]
            ys = [p[1] for p in coords.points]
            sx, sy = size[0] / coords.system.width, size[1] / coords.system.height
            bbox = (min(xs) * sx, min(ys) * sy, max(xs) * sx, max(ys) * sy)
        text = (it.text or "").strip()
        if kind == Kind.TABLE and getattr(md, "text_as_html", None):
            rows = _html_rows(md.text_as_html)
            if len(rows) >= 2:
                parsed.elements.append(Element(kind=Kind.TABLE, text=rows_markdown(rows), page=page,
                                               bbox=bbox, table_rows=rows,
                                               meta={"table_idx": len(parsed.tables)}))  # fmt: skip
                parsed.tables.append(rows_to_table(rows, label=f"{path.name} p.{page}", page=page))
                continue
        if text:
            parsed.elements.append(Element(kind=kind if kind != Kind.FIGURE else Kind.PARAGRAPH,
                                           text=text, page=page, bbox=bbox,
                                           heading_level=1 if kind == Kind.HEADING else None))  # fmt: skip
    parsed.page_count = max((e.page or 0 for e in parsed.elements), default=0) or None
    return parsed
