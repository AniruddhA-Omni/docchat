"""Image parser: OCR lines with boxes (scans, screenshots) plus a figure element for the vision agent."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from docchat.ingest.context import ParseContext
from docchat.ingest.ocr import group_lines, union_bbox
from docchat.schemas import Element, Kind, ParsedDocument

MAX_OCR_SIDE = 3000


def parse_image(path: Path, ctx: ParseContext) -> ParsedDocument:
    parsed = ParsedDocument(doc_id=ctx.doc_id, file_name=path.name, path=path, format="image",
                            parser="image", page_count=1)  # fmt: skip
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    # Normalized copy (EXIF-rotated RGB PNG) used for citation previews and the vision agent.
    ctx.render_dir.mkdir(parents=True, exist_ok=True)
    normalized = ctx.render_dir / "image.png"
    image.save(normalized)

    scale = 1.0
    ocr_input = image
    if max(image.size) > MAX_OCR_SIDE:
        scale = MAX_OCR_SIDE / max(image.size)
        ocr_input = image.resize((round(image.width * scale), round(image.height * scale)))

    ocr_lines = []
    try:
        engine = ctx.ocr()
        ocr_lines = engine.recognize(ocr_input)
        parsed.ocr_engine = engine.name
    except Exception as exc:
        parsed.warnings.append(f"OCR failed ({exc})")

    for block in group_lines(ocr_lines):
        box = tuple(v / scale for v in union_bbox([ln.bbox for ln in block]))
        conf = sum(ln.conf for ln in block) / len(block)
        parsed.elements.append(Element(kind=Kind.OCR_TEXT, text="\n".join(ln.text for ln in block),
                                       page=1, bbox=box, meta={"ocr_conf": round(conf, 3)}))  # fmt: skip

    all_text = " ".join(ln.text for ln in ocr_lines)
    summary = f"Image {path.name} ({image.width}x{image.height} px)."
    if all_text:
        summary += f" Text in image: {all_text[:1500]}"
    parsed.elements.insert(0, Element(
        kind=Kind.FIGURE, text=summary, page=1, bbox=(0, 0, image.width, image.height),
        image_path=str(normalized), meta={"ocr_text": all_text, "standalone": True},
    ))  # fmt: skip
    return parsed
