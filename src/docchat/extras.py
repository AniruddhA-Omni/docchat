"""Detect which optional parser / OCR backends (uv extras) are installed."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from functools import cache

# backend name -> (import name, uv extra that provides it)
PDF_PARSERS: dict[str, tuple[str, str | None]] = {
    "pymupdf": ("pymupdf", None),
    "docling": ("docling", "docling"),
    "marker": ("marker", None),  # external CLI in its own uv tool env, see backend_available
    "unstructured": ("unstructured", "unstructured"),
}
OCR_BACKENDS: dict[str, tuple[str, str | None]] = {
    "rapidocr": ("rapidocr", None),
    "tesseract": ("pytesseract", "tesseract"),
    "paddleocr": ("paddleocr", "paddleocr"),
}


# Backends with native extensions: a package can be installed but unusable (missing DLLs).
_NATIVE = {"pymupdf": ("pymupdf", None), "rapidocr": ("rapidocr", "RapidOCR")}


@cache
def _importable(module: str) -> bool:
    if importlib.util.find_spec(module) is None:
        return False
    if module in _NATIVE:
        name, attr = _NATIVE[module]
        try:
            mod = importlib.import_module(name)
            if attr:
                getattr(mod, attr)
        except (ImportError, OSError):
            return False
    return True


def backend_available(kind: str, name: str) -> bool:
    if kind == "pdf" and name == "marker":
        # Marker pins old shared dependencies, so it runs as an isolated uv tool (pdf_marker.py).
        return shutil.which("marker_single") is not None
    table = PDF_PARSERS if kind == "pdf" else OCR_BACKENDS
    module, _ = table[name]
    if not _importable(module):
        return False
    # pytesseract is only a wrapper; the tesseract binary must be on PATH too.
    return not (name == "tesseract" and shutil.which("tesseract") is None)


def install_hint(kind: str, name: str) -> str:
    if kind == "pdf" and name == "marker":
        return "uv tool install marker-pdf  (isolated environment; GPU strongly recommended)"
    table = PDF_PARSERS if kind == "pdf" else OCR_BACKENDS
    _, extra = table[name]
    hint = f"uv sync --extra {extra}" if extra else "uv sync"
    if name == "tesseract":
        hint += "  (plus the binary: winget install UB-Mannheim.TesseractOCR)"
    return hint
