"""High-fidelity PDF backend: IBM Docling (extra ``docling``; MIT).

Docling runs layout + TableFormer models (PyTorch; GPU recommended for large PDFs) and returns a
structured document: section headers, paragraphs, list items, captions, tables with cell
structure and pictures, each with page provenance. Boxes are converted to a top-left origin so
citations highlight the same way as with the PyMuPDF backend.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from docchat.ingest.context import ParseContext
from docchat.ingest.tables import rows_markdown, rows_to_table
from docchat.schemas import Element, Kind, ParsedDocument


@cache
def _converter(ocr_backend: str):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_table_structure = True
    opts.do_ocr = True
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    try:  # match the configured OCR engine where Docling supports it
        from docling.datamodel.pipeline_options import RapidOcrOptions, TesseractCliOcrOptions

        opts.ocr_options = (
            TesseractCliOcrOptions() if ocr_backend == "tesseract" else RapidOcrOptions()
        )
    except ImportError:
        pass
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _prov(item, doc) -> tuple[int | None, tuple | None]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None, None
    p = prov[0]
    page_no = p.page_no
    bbox = None
    try:
        height = doc.pages[page_no].size.height
        b = p.bbox.to_top_left_origin(page_height=height)
        bbox = (b.l, b.t, b.r, b.b)
    except (AttributeError, KeyError):
        pass
    return page_no, bbox


def parse_pdf(path: Path, ctx: ParseContext) -> ParsedDocument:
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem

    ctx.report(f"{path.name}: running Docling layout analysis")
    result = _converter(ctx.settings.ocr_backend).convert(str(path))
    doc = result.document
    parsed = ParsedDocument(doc_id=ctx.doc_id, file_name=path.name, path=path, format="pdf",
                            parser="docling", page_count=len(doc.pages))  # fmt: skip
    n_fig = 0
    for item, level in doc.iterate_items():
        page, bbox = _prov(item, doc)
        if isinstance(item, SectionHeaderItem):
            parsed.elements.append(Element(kind=Kind.HEADING, text=item.text, page=page, bbox=bbox,
                                           heading_level=getattr(item, "level", level) or 1))  # fmt: skip
        elif isinstance(item, TableItem):
            df = item.export_to_dataframe()
            if df.empty:
                continue
            rows = [
                [str(c) for c in df.columns],
                *[[str(v) for v in r] for r in df.itertuples(index=False)],
            ]
            caption = item.caption_text(doc) or ""
            parsed.elements.append(Element(kind=Kind.TABLE, text=(caption + "\n" if caption else "")
                                           + rows_markdown(rows), page=page, bbox=bbox, table_rows=rows,
                                           meta={"table_idx": len(parsed.tables), "caption": caption}))  # fmt: skip
            table = rows_to_table(rows, label=f"{path.name} p.{page}", page=page)
            table.title = caption or f"Table on page {page} of {path.name}"
            parsed.tables.append(table)
        elif isinstance(item, PictureItem):
            image = item.get_image(doc)
            if image is None:
                continue
            n_fig += 1
            ctx.render_dir.mkdir(parents=True, exist_ok=True)
            out = ctx.render_dir / f"p{page or 0:04d}_fig{n_fig}.png"
            image.save(out)
            caption = item.caption_text(doc) or ""
            parsed.elements.append(Element(kind=Kind.FIGURE, page=page, bbox=bbox, image_path=str(out),
                                           text=f"{caption}\nFigure on page {page}.".strip(),
                                           meta={"caption": caption}))  # fmt: skip
        elif isinstance(item, TextItem):
            label = str(getattr(item, "label", "text")).lower()
            kind = {"page_header": Kind.HEADER, "page_footer": Kind.FOOTER, "caption": Kind.CAPTION,
                    "list_item": Kind.LIST, "title": Kind.HEADING, "code": Kind.CODE,
                    "footnote": Kind.PARAGRAPH}.get(label.split(".")[-1], Kind.PARAGRAPH)  # fmt: skip
            if item.text.strip():
                parsed.elements.append(Element(kind=kind, text=item.text.strip(), page=page, bbox=bbox,
                                               heading_level=1 if kind == Kind.HEADING else None))  # fmt: skip
    parsed.ocr_engine = ctx.settings.ocr_backend
    return parsed
