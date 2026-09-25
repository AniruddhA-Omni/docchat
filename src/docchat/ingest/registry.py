"""Map a file to a parser backend. Backends are imported lazily, and the chosen PDF backend
falls back along ``chosen -> pymupdf`` when it is missing or fails, so one broken optional
dependency never blocks ingestion.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path

from docchat.extras import backend_available
from docchat.ingest.context import ParseContext
from docchat.ingest.detect import Format, detect_format
from docchat.schemas import ParsedDocument

log = logging.getLogger(__name__)

ParserFn = Callable[[Path, ParseContext], ParsedDocument]

# format -> "module:function"
_DEFAULT: dict[Format, str] = {
    Format.PDF: "docchat.ingest.parsers.pdf_pymupdf:parse_pdf",
    Format.DOCX: "docchat.ingest.parsers.office:parse_docx",
    Format.PPTX: "docchat.ingest.parsers.office:parse_pptx",
    Format.TEXT: "docchat.ingest.parsers.text:parse_text",
    Format.MARKDOWN: "docchat.ingest.parsers.text:parse_markdown",
    Format.TABULAR: "docchat.ingest.parsers.tabular:parse_tabular",
    Format.IMAGE: "docchat.ingest.parsers.image:parse_image",
    Format.CODE: "docchat.ingest.parsers.code:parse_code",
}
_PDF_BACKENDS: dict[str, str] = {
    "pymupdf": _DEFAULT[Format.PDF],
    "docling": "docchat.ingest.parsers.pdf_docling:parse_pdf",
    "marker": "docchat.ingest.parsers.pdf_marker:parse_pdf",
    "unstructured": "docchat.ingest.parsers.pdf_unstructured:parse_pdf",
}


class UnsupportedFileError(ValueError):
    pass


def _load(target: str) -> ParserFn:
    module, func = target.split(":")
    return getattr(importlib.import_module(module), func)


def parse_file(path: Path, ctx: ParseContext) -> ParsedDocument:
    fmt = detect_format(path)
    if fmt is None:
        raise UnsupportedFileError(f"Unsupported file type: {path.suffix or path.name}")
    targets = [_DEFAULT[fmt]]
    if fmt == Format.PDF:
        chosen = ctx.settings.pdf_parser
        if chosen != "pymupdf" and backend_available("pdf", chosen):
            targets.insert(0, _PDF_BACKENDS[chosen])
        elif chosen != "pymupdf":
            log.warning("PDF backend %s is not installed; using pymupdf", chosen)
    errors = []
    for target in targets:
        try:
            parsed = _load(target)(path, ctx)
        except UnsupportedFileError:
            raise
        except Exception as exc:  # noqa: BLE001 - try the next backend
            log.exception("parser %s failed on %s", target, path.name)
            errors.append(f"{target.split(':')[0].rsplit('.', 1)[-1]}: {exc}")
            continue
        parsed.warnings.extend(f"fallback after {e}" for e in errors)
        return parsed
    raise RuntimeError(f"Could not parse {path.name}: " + "; ".join(errors))
