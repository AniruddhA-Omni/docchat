"""RapidOCR engine (ONNX Runtime, CPU by default)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from docchat.ingest.ocr import OcrLine


class RapidEngine:
    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr import RapidOCR

        self._engine = RapidOCR()

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        out = self._engine(np.asarray(image.convert("RGB")))
        if out.boxes is None or out.txts is None:
            return []
        lines = []
        for box, text, score in zip(out.boxes, out.txts, out.scores, strict=False):
            xs, ys = box[:, 0], box[:, 1]
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            if text.strip():
                lines.append(OcrLine(text=text.strip(), bbox=bbox, conf=float(score)))
        return lines
