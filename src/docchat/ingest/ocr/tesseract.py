"""Tesseract engine (optional extra `tesseract`; needs tesseract.exe on PATH)."""

from __future__ import annotations

from collections import defaultdict

from PIL import Image

from docchat.ingest.ocr import OcrLine


class TesseractEngine:
    name = "tesseract"

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        import pytesseract

        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i, text in enumerate(data["text"]):
            if text.strip() and float(data["conf"][i]) >= 0:
                words[(data["block_num"][i], data["par_num"][i], data["line_num"][i])].append(i)
        lines = []
        for idx in words.values():
            x0 = min(data["left"][i] for i in idx)
            y0 = min(data["top"][i] for i in idx)
            x1 = max(data["left"][i] + data["width"][i] for i in idx)
            y1 = max(data["top"][i] + data["height"][i] for i in idx)
            conf = sum(float(data["conf"][i]) for i in idx) / len(idx) / 100
            text = " ".join(data["text"][i] for i in idx)
            lines.append(OcrLine(text=text, bbox=(x0, y0, x1, y1), conf=conf))
        return lines
