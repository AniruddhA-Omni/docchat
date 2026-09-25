"""Pluggable OCR engines. RapidOCR (PaddleOCR models on ONNX) needs no system install."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache
from typing import Protocol

from PIL import Image

from docchat.extras import backend_available
from docchat.schemas import BBox

log = logging.getLogger(__name__)


@dataclass
class OcrLine:
    text: str
    bbox: BBox  # pixels in the input image
    conf: float


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: Image.Image) -> list[OcrLine]: ...


@cache
def get_ocr_engine(name: str) -> OcrEngine:
    """Return the requested engine, falling back to RapidOCR if it is not installed."""
    if name != "rapidocr" and not backend_available("ocr", name):
        log.warning("OCR backend %s not available, using rapidocr", name)
        name = "rapidocr"
    if name == "tesseract":
        from docchat.ingest.ocr.tesseract import TesseractEngine

        return TesseractEngine()
    if name == "paddleocr":
        from docchat.ingest.ocr.paddle import PaddleEngine

        return PaddleEngine()
    from docchat.ingest.ocr.rapid import RapidEngine

    return RapidEngine()


def group_lines(lines: list[OcrLine], gap_factor: float = 1.2) -> list[list[OcrLine]]:
    """Group OCR lines into paragraph-like blocks by vertical gaps (lines sorted top-down)."""
    ordered = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    blocks: list[list[OcrLine]] = []
    for line in ordered:
        if blocks:
            prev = blocks[-1][-1]
            height = max(prev.bbox[3] - prev.bbox[1], 1.0)
            if line.bbox[1] - prev.bbox[3] <= gap_factor * height:
                blocks[-1].append(line)
                continue
        blocks.append([line])
    return blocks


def union_bbox(boxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
