"""Per-document state handed to every parser backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from docchat.config import Settings
from docchat.ingest.ocr import OcrEngine, get_ocr_engine


@dataclass
class ParseContext:
    settings: Settings
    doc_id: str
    render_dir: Path  # figure crops / page renders for citations and the vision agent
    progress: Callable[[str], None] | None = None

    def report(self, message: str) -> None:
        if self.progress:
            self.progress(message)

    def ocr(self) -> OcrEngine:
        return get_ocr_engine(self.settings.ocr_backend)
