"""Visual citation previews: the cited PDF page with the supporting sentence highlighted, or the
cited image with its OCR boxes drawn. Rendered PNGs are cached under ``data/renders/_cite``."""

from __future__ import annotations

import hashlib
import io
import logging
import re
from pathlib import Path

from PIL import Image, ImageDraw

from docchat.agents.types import Citation

log = logging.getLogger(__name__)
PREVIEW_DPI = 110
HIGHLIGHT = (1.0, 0.85, 0.1)


def _cache_path(cache_dir: Path, *parts) -> Path:
    key = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.png"


def _phrases(quote: str) -> list[str]:
    """Search phrases from most to least specific (PDF text may break lines differently)."""
    quote = re.sub(r"\s+", " ", quote).strip()
    words = quote.split()
    out = [quote[:120]]
    if len(words) > 8:
        out += [" ".join(words[:8]), " ".join(words[-8:])]
    if len(words) > 4:
        out.append(" ".join(words[:4]))
    return [p for p in out if len(p) >= 6]


def render_pdf_citation(pdf_path: Path, cite: Citation, cache_dir: Path) -> bytes | None:
    if not cite.page or not pdf_path.exists():
        return None
    out = _cache_path(cache_dir, pdf_path, cite.page, cite.quote, cite.bboxes)
    if out.exists():
        return out.read_bytes()
    try:
        import pymupdf
    except (ImportError, OSError) as exc:  # native library missing
        log.info("PDF preview unavailable: %s", exc)
        return None
    try:
        with pymupdf.open(pdf_path) as doc:
            page = doc[cite.page - 1]
            rects = []
            for phrase in _phrases(cite.quote) if cite.quote else []:
                rects = page.search_for(phrase)
                if rects:
                    break
            if not rects:
                rects = [pymupdf.Rect(b[1:5]) for b in cite.bboxes if int(b[0]) == cite.page]
            for r in rects:
                page.draw_rect(r + (-2, -1, 2, 1), color=HIGHLIGHT, fill=HIGHLIGHT,
                               fill_opacity=0.35, width=1.2, overlay=True)  # fmt: skip
            clip = None
            if rects:  # crop to the highlighted region with generous context
                area = rects[0]
                for r in rects[1:]:
                    area |= r
                clip = pymupdf.Rect(page.rect.x0, max(page.rect.y0, area.y0 - 160),
                                    page.rect.x1, min(page.rect.y1, area.y1 + 160))  # fmt: skip
            pix = page.get_pixmap(dpi=PREVIEW_DPI, clip=clip)
            data = pix.tobytes("png")
    except Exception as exc:
        log.warning("PDF preview failed for %s p.%s: %s", pdf_path.name, cite.page, exc)
        return None
    out.write_bytes(data)
    return data


def render_image_citation(
    image_path: Path, boxes: list[list[float]], cache_dir: Path
) -> bytes | None:
    if not image_path.exists():
        return None
    out = _cache_path(cache_dir, image_path, boxes)
    if out.exists():
        return out.read_bytes()
    with Image.open(image_path) as im:
        img = im.convert("RGB")
    if boxes:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for b in boxes:
            x0, y0, x1, y1 = b[-4:]
            if (x1 - x0) * (y1 - y0) >= 0.9 * img.width * img.height:
                continue  # the whole-image box of a figure
            draw.rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), fill=(255, 215, 0, 70),
                           outline=(230, 150, 0, 255), width=3)  # fmt: skip
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img.thumbnail((1400, 1400))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out.write_bytes(buf.getvalue())
    return buf.getvalue()


def citation_preview(cite: Citation, source_path: Path | None, cache_dir: Path) -> bytes | None:
    """Best visual preview for a citation, or None (the UI then shows the text snippet)."""
    try:
        if cite.image_path and Path(cite.image_path).exists() and cite.file_type in ("image", ""):
            return render_image_citation(Path(cite.image_path), cite.bboxes, cache_dir)
        if source_path and source_path.suffix.lower() == ".pdf":
            return render_pdf_citation(source_path, cite, cache_dir)
        if cite.image_path and Path(cite.image_path).exists():
            return render_image_citation(Path(cite.image_path), [], cache_dir)
    except Exception as exc:  # previews are best effort
        log.warning("citation preview failed: %s", exc)
    return None
