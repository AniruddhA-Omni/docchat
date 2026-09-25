"""Default PDF backend: PyMuPDF layout analysis with per-page OCR fallback.

For each page we extract text blocks with font statistics (-> headings, captions, lists,
page headers/footers), ruled tables (``find_tables``), embedded figures (cropped to PNG for the
vision agent) and, when a page has no usable text layer (scans), OCR lines with boxes.
All coordinates are PDF points with a top-left origin, so citations can highlight regions.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf
from PIL import Image

from docchat.ingest.context import ParseContext
from docchat.ingest.ocr import group_lines, union_bbox
from docchat.ingest.tables import rows_markdown, rows_to_table
from docchat.schemas import BBox, Element, Kind, ParsedDocument

OCR_DPI = 200
FIGURE_DPI = 150
MARGIN = 0.07  # top/bottom fraction of the page treated as header/footer zone
MIN_TEXT_CHARS = 50
CAPTION_RE = re.compile(r"^(figure|fig\.|table|chart|exhibit|image)\s*\d+", re.I)
BULLET_RE = re.compile(r"^\s*([•●▪◦\-–*]|\d+[.)]|[a-z][.)])\s+")


def _pix_to_image(pix: pymupdf.Pixmap) -> Image.Image:
    mode = "RGBA" if pix.alpha else "RGB"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")


def _body_font_size(doc: pymupdf.Document, max_pages: int = 30) -> float:
    sizes: Counter[float] = Counter()
    for page in doc.pages(0, min(max_pages, doc.page_count)):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _center_inside(bbox: BBox, rect: pymupdf.Rect) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1


def _block_text(block: dict) -> tuple[str, float, bool]:
    """Return reflowed text, max font size and whether every span is bold."""
    lines, max_size, all_bold = [], 0.0, True
    for line in block["lines"]:
        spans = [s for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        lines.append("".join(s["text"] for s in line["spans"]).strip())
        max_size = max(max_size, *(s["size"] for s in spans))
        all_bold &= all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
    return "\n".join(lines), max_size, all_bold and bool(lines)


def _reflow(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line breaks
    return re.sub(r"\s*\n\s*", " ", text).strip()


def parse_pdf(path: Path, ctx: ParseContext) -> ParsedDocument:
    parsed = ParsedDocument(
        doc_id=ctx.doc_id, file_name=path.name, path=path, format="pdf", parser="pymupdf"
    )
    with pymupdf.open(path) as doc:
        parsed.page_count = doc.page_count
        body_size = _body_font_size(doc)
        heading_sizes: set[float] = set()
        pending_headings: list[Element] = []

        for page in doc:
            pno = page.number + 1
            ctx.report(f"{path.name}: page {pno}/{doc.page_count}")
            width, height = page.rect.width, page.rect.height
            page_elements: list[Element] = []

            # --- ruled tables ----------------------------------------------------------------
            table_rects: list[pymupdf.Rect] = []
            try:
                found = page.find_tables().tables
            except Exception as exc:  # table detection is best effort
                parsed.warnings.append(f"p.{pno}: table detection failed ({exc})")
                found = []
            for t in found:
                rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in t.extract()]
                rows = [r for r in rows if any(r)]
                if len(rows) < 2 or len(rows[0]) < 2:
                    continue
                rect = pymupdf.Rect(t.bbox)
                table_rects.append(rect)
                page_elements.append(Element(kind=Kind.TABLE, text=rows_markdown(rows), page=pno,
                                             bbox=tuple(rect), table_rows=rows,
                                             meta={"table_idx": len(parsed.tables)}))  # fmt: skip
                table = rows_to_table(rows, label=f"{path.name} p.{pno}", page=pno)
                table.title = f"Table on page {pno} of {path.name}"
                parsed.tables.append(table)

            # --- text blocks -----------------------------------------------------------------
            text_chars = 0
            for block in page.get_text("dict", sort=True)["blocks"]:
                if block["type"] != 0:
                    continue
                bbox = tuple(block["bbox"])
                if any(_center_inside(bbox, r) for r in table_rects):
                    continue
                raw, size, bold = _block_text(block)
                if not raw:
                    continue
                text_chars += len(raw)
                in_margin = bbox[3] < height * MARGIN or bbox[1] > height * (1 - MARGIN)
                one_liner = len(raw) <= 150 and raw.count("\n") <= 1
                if in_margin and len(raw) <= 120:
                    kind = Kind.HEADER if bbox[3] < height * MARGIN else Kind.FOOTER
                    page_elements.append(Element(kind=kind, text=_reflow(raw), page=pno, bbox=bbox))
                elif one_liner and (
                    size >= body_size * 1.2 or (bold and size >= body_size and raw[-1] != ".")
                ):
                    el = Element(kind=Kind.HEADING, text=_reflow(raw), page=pno, bbox=bbox,
                                 meta={"font_size": round(size, 1)})  # fmt: skip
                    heading_sizes.add(round(size, 1))
                    pending_headings.append(el)
                    page_elements.append(el)
                elif CAPTION_RE.match(raw) and len(raw) <= 300:
                    page_elements.append(Element(kind=Kind.CAPTION, text=_reflow(raw), page=pno,
                                                 bbox=bbox))  # fmt: skip
                elif BULLET_RE.match(raw):
                    page_elements.append(Element(kind=Kind.LIST, text=raw, page=pno, bbox=bbox))
                else:
                    page_elements.append(Element(kind=Kind.PARAGRAPH, text=_reflow(raw), page=pno,
                                                 bbox=bbox))  # fmt: skip

            # --- scanned page -> OCR -----------------------------------------------------------
            images = [i for i in page.get_image_info(xrefs=True) if i.get("bbox")]
            page_area = width * height
            coverage = sum(pymupdf.Rect(i["bbox"]).get_area() for i in images) / page_area
            ocr_done = False
            if text_chars < MIN_TEXT_CHARS and (coverage > 0.3 or not text_chars):
                ocr_done = _ocr_page(page, pno, ctx, parsed, page_elements)

            # --- figures -----------------------------------------------------------------------
            for k, info in enumerate(images, 1):
                rect = pymupdf.Rect(info["bbox"]) & page.rect
                if rect.is_empty or rect.get_area() < 0.02 * page_area:
                    continue
                if ocr_done and rect.get_area() > 0.6 * page_area:
                    continue  # the page scan itself, already OCR'd
                page_elements.append(_figure(page, pno, k, rect, ctx, parsed))

            page_elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]) if e.bbox else (0, 0))
            _attach_captions(page_elements)
            parsed.elements.extend(page_elements)

        # Map heading font sizes to levels (largest = 1).
        levels = {s: i + 1 for i, s in enumerate(sorted(heading_sizes, reverse=True)[:4])}
        for el in pending_headings:
            el.heading_level = levels.get(el.meta["font_size"], 4)
    return parsed


def _ocr_page(
    page, pno: int, ctx: ParseContext, parsed: ParsedDocument, out: list[Element]
) -> bool:
    try:
        engine = ctx.ocr()
    except Exception as exc:
        parsed.warnings.append(f"p.{pno}: OCR unavailable ({exc})")
        return False
    image = _pix_to_image(page.get_pixmap(dpi=OCR_DPI))
    lines = engine.recognize(image)
    parsed.ocr_engine = engine.name
    scale = 72 / OCR_DPI
    for block in group_lines(lines):
        box = union_bbox([ln.bbox for ln in block])
        out.append(Element(
            kind=Kind.OCR_TEXT,
            text="\n".join(ln.text for ln in block),
            page=pno,
            bbox=tuple(v * scale for v in box),
            meta={"ocr_conf": round(sum(ln.conf for ln in block) / len(block), 3)},
        ))  # fmt: skip
    return bool(lines)


def _figure(page, pno: int, k: int, rect, ctx: ParseContext, parsed: ParsedDocument) -> Element:
    ctx.render_dir.mkdir(parents=True, exist_ok=True)
    crop_path = ctx.render_dir / f"p{pno:04d}_fig{k}.png"
    pix = page.get_pixmap(dpi=FIGURE_DPI, clip=rect)
    pix.save(crop_path)
    ocr_text = ""
    try:
        engine = ctx.ocr()
        ocr_text = " ".join(ln.text for ln in engine.recognize(_pix_to_image(pix)))
        parsed.ocr_engine = engine.name
    except Exception as exc:
        parsed.warnings.append(f"p.{pno}: figure OCR failed ({exc})")
    text = f"Figure on page {pno}."
    if ocr_text:
        text += f" Text in figure: {ocr_text}"
    return Element(kind=Kind.FIGURE, text=text, page=pno, bbox=tuple(rect),
                   image_path=str(crop_path), meta={"ocr_text": ocr_text})  # fmt: skip


def _attach_captions(elements: list[Element], max_gap: float = 60.0) -> None:
    """Prefix each figure/table with the nearest caption on the same page."""
    captions = [e for e in elements if e.kind == Kind.CAPTION and e.bbox]
    for el in elements:
        if el.kind not in (Kind.FIGURE, Kind.TABLE) or not el.bbox or not captions:
            continue
        near = min(captions, key=lambda c: min(abs(c.bbox[1] - el.bbox[3]),
                                               abs(el.bbox[1] - c.bbox[3])))  # fmt: skip
        gap = min(abs(near.bbox[1] - el.bbox[3]), abs(el.bbox[1] - near.bbox[3]))
        if gap <= max_gap:
            el.meta["caption"] = near.text
            el.text = f"{near.text}\n{el.text}"
