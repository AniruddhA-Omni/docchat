from pathlib import Path

import pytest

from docchat.ingest.chunking import chunk_document
from docchat.ingest.parsers.code import parse_code
from docchat.ingest.parsers.office import parse_docx, parse_pptx
from docchat.ingest.parsers.tabular import parse_tabular
from docchat.ingest.parsers.text import parse_markdown, parse_text
from docchat.schemas import Kind
from tests.conftest import requires_pymupdf, requires_rapidocr


def _texts(parsed, kind):
    return [e for e in parsed.elements if e.kind == kind]


def test_docx_headings_pages_and_tables(corpus: Path, parse_ctx):
    p = parse_docx(corpus / "orion_amr_v2_prd.docx", parse_ctx)
    headings = {e.text: e for e in _texts(p, Kind.HEADING)}
    assert headings["3.1 Payload and speed"].heading_level == 2
    payload = next(e for e in p.elements if "150 kg" in e.text)
    assert payload.page == 3  # explicit page breaks are tracked
    assert len(p.tables) == 1 and len(p.tables[0].df) == 6


def test_pptx_notes_and_exact_chart_data(corpus: Path, parse_ctx):
    p = parse_pptx(corpus / "board_update_q4.pptx", parse_ctx)
    notes = _texts(p, Kind.NOTES)
    assert notes and "Singapore" in notes[0].text and notes[0].page == 4
    chart = next(t for t in p.tables if "Chart data" in (t.title or ""))
    assert (
        dict(zip(chart.df["category"], chart.df["units_shipped"], strict=True))["Vega Vision Kit"]
        == 2310
    )


def test_markdown_line_numbers(corpus: Path, parse_ctx):
    p = parse_markdown(corpus / "engineering_handbook.md", parse_ctx)
    freeze = next(e for e in p.elements if "frozen" in e.text)
    lines = (corpus / "engineering_handbook.md").read_text(encoding="utf-8").split("\n")
    assert "frozen" in lines[freeze.line_start - 1]


def test_text_headings_and_lists(corpus: Path, parse_ctx):
    p = parse_text(corpus / "meeting_notes_2025-11-03.txt", parse_ctx)
    assert [e.text for e in _texts(p, Kind.HEADING)] == ["Decisions", "Action items"]


def test_python_symbols_include_methods(corpus: Path, parse_ctx):
    p = parse_code(corpus / "inventory_service.py", parse_ctx)
    symbols = {s["symbol"]: s for s in p.symbols}
    assert "InventoryService.list_items" in symbols
    src = (corpus / "inventory_service.py").read_text(encoding="utf-8").split("\n")
    assert "def list_items" in src[symbols["InventoryService.list_items"]["line_start"] - 1]


def test_regex_symbols_for_other_languages(tmp_path, parse_ctx):
    f = tmp_path / "app.ts"
    f.write_text("export function add(a: number, b: number) {\n  return a + b;\n}\n\n"
                 "export class Cart {\n  items = [];\n}\n", encoding="utf-8")  # fmt: skip
    p = parse_code(f, parse_ctx)
    assert {s["symbol"] for s in p.symbols} >= {"add", "Cart"}


def test_xlsx_sheets(corpus: Path, parse_ctx):
    p = parse_tabular(corpus / "sales_2025.xlsx", parse_ctx)
    by_sheet = {t.sheet: t for t in p.tables}
    assert set(by_sheet) == {"Orders", "Regional Summary", "Targets"}
    summary = by_sheet["Regional Summary"]
    assert "h2_revenue_usd" in summary.df.columns
    assert summary.df["_is_total"].sum() == 1
    assert len(by_sheet["Orders"].df) == 400


def test_csv_types(corpus: Path, parse_ctx):
    [t] = parse_tabular(corpus / "support_tickets.csv", parse_ctx).tables
    assert t.df["ticket_id"].iloc[0] == "T-0001"
    assert str(t.df["resolution_hours"].dtype) == "float64"


def test_chunks_respect_limits_and_sections(corpus: Path, parse_ctx, settings):
    for name, fn in [("orion_amr_v2_prd.docx", parse_docx), ("board_update_q4.pptx", parse_pptx),
                     ("engineering_handbook.md", parse_markdown), ("inventory_service.py", parse_code)]:  # fmt: skip
        chunks = chunk_document(fn(corpus / name, parse_ctx), settings)
        assert chunks, name
        for c in chunks:
            assert c.token_count <= settings.chunk_max_tokens + 60, (name, c.token_count)
            assert c.file_name == name
            assert c.page_start or c.line_start, (name, c.text[:40])


def test_long_paragraph_and_table_are_split(parse_ctx, settings):
    from docchat.ingest.tables import rows_markdown
    from docchat.schemas import Element, ParsedDocument

    long_text = " ".join(f"Sentence number {i} talks about robots." for i in range(400))
    rows = [["id", "name"]] + [[str(i), f"name {i} " * 5] for i in range(300)]
    parsed = ParsedDocument(
        doc_id="d", file_name="f.pdf", path=Path("f.pdf"), format="pdf", parser="x"
    )
    parsed.elements = [
        Element(kind=Kind.HEADING, text="Intro", page=1, heading_level=1),
        Element(kind=Kind.PARAGRAPH, text=long_text, page=1),
        Element(kind=Kind.TABLE, text=rows_markdown(rows), page=2, table_rows=rows),
    ]
    chunks = chunk_document(parsed, settings)
    prose = [c for c in chunks if c.kind == Kind.PARAGRAPH]
    tables = [c for c in chunks if c.kind == Kind.TABLE]
    assert len(prose) > 3 and len(tables) > 3
    assert all(c.section_path == ["Intro"] for c in chunks)
    assert all(t.text.split("\n")[1].startswith("| id | name |") for t in tables)
    assert all(c.token_count <= settings.chunk_max_tokens + 60 for c in chunks)


@requires_pymupdf
def test_pdf_pages_tables_and_figures(corpus: Path, parse_ctx):
    from docchat.ingest.parsers.pdf_pymupdf import parse_pdf

    p = parse_pdf(corpus / "acme_annual_report_2025.pdf", parse_ctx)
    assert p.page_count == 10
    assert next(e for e in p.elements if "48.6" in e.text).page == 1
    assert next(e for e in p.elements if "412" in e.text).page == 5
    region = next(t for t in p.tables if t.page == 2)
    assert "APAC" in region.df.iloc[:, 0].tolist()
    assert any(e.kind == Kind.FIGURE and e.page == 3 for e in p.elements)
    assert not any("Annual Report FY2025" in e.text for e in p.elements if e.kind == Kind.PARAGRAPH
                   and e.page and e.page > 1)  # footers are not body text  # fmt: skip


@requires_pymupdf
@requires_rapidocr
def test_scanned_pdf_is_ocrd(corpus: Path, parse_ctx):
    from docchat.ingest.parsers.pdf_pymupdf import parse_pdf

    p = parse_pdf(corpus / "supplier_agreement_scanned.pdf", parse_ctx)
    text = " ".join(e.text for e in p.elements if e.kind == Kind.OCR_TEXT)
    assert "1,250,000" in text or "1250000" in text
    assert p.ocr_engine == "rapidocr"


@requires_rapidocr
def test_image_ocr_boxes(corpus: Path, parse_ctx):
    from docchat.ingest.parsers.image import parse_image

    p = parse_image(corpus / "invoice_scan.png", parse_ctx)
    ocr = [e for e in p.elements if e.kind == Kind.OCR_TEXT]
    assert any("18,450" in e.text for e in ocr)
    assert all(e.bbox for e in ocr)
    assert p.elements[0].kind == Kind.FIGURE and Path(p.elements[0].image_path).exists()


@pytest.mark.parametrize("name", ["a.pdf", "b.docx", "c.png", "d.xlsx", "e.py"])
def test_registry_resolves_every_format(name):
    from docchat.ingest.detect import detect_format
    from docchat.ingest.registry import _DEFAULT

    assert detect_format(name) in _DEFAULT
