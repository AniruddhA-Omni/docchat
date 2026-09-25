"""PaddleOCR engine (optional extra `paddleocr`)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from docchat.ingest.ocr import OcrLine


class PaddleEngine:
    name = "paddleocr"

    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        lines = []
        for res in self._engine.predict(np.asarray(image.convert("RGB"))):
            for text, score, box in zip(
                res["rec_texts"], res["rec_scores"], res["rec_boxes"], strict=False
            ):
                x0, y0, x1, y1 = (float(v) for v in box)
                if text.strip():
                    lines.append(OcrLine(text=text.strip(), bbox=(x0, y0, x1, y1), conf=score))
        return lines
