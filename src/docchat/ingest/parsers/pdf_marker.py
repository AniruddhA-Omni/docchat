"""Marker PDF backend, run as an isolated tool: ``uv tool install marker-pdf``.

Marker pins old versions of shared dependencies (e.g. ``pillow<11``), so instead of forcing them
on the whole project it lives in its own uv tool environment and is invoked through its
``marker_single`` CLI. Code is Apache-2.0 (v2); the model weights use a modified OpenRAIL-M
licence (free for research, personal use and small companies). A GPU is strongly recommended.
Marker produces paginated Markdown, which is parsed page by page.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from docchat.ingest.context import ParseContext
from docchat.ingest.parsers.text import markdown_into
from docchat.schemas import ParsedDocument

_PAGE_SPLIT = re.compile(r"\n*\{(\d+)\}-{20,}\n*")
TIMEOUT_S = 1800


def parse_pdf(path: Path, ctx: ParseContext) -> ParsedDocument:
    exe = shutil.which("marker_single")
    if exe is None:
        raise RuntimeError("marker_single not found; install with `uv tool install marker-pdf`")
    ctx.report(f"{path.name}: running Marker (isolated tool)")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [exe, str(path), "--output_dir", tmp, "--output_format", "markdown", "--paginate_output"],
            check=True, capture_output=True, timeout=TIMEOUT_S,
        )  # fmt: skip
        md_files = list(Path(tmp).rglob("*.md"))
        if not md_files:
            raise RuntimeError("Marker produced no Markdown output")
        markdown = md_files[0].read_text(encoding="utf-8")
        ctx.render_dir.mkdir(parents=True, exist_ok=True)
        for img in md_files[0].parent.glob("*.[pj][pn]g"):
            shutil.copy2(img, ctx.render_dir / img.name)

    parsed = ParsedDocument(doc_id=ctx.doc_id, file_name=path.name, path=path, format="pdf",
                            parser="marker")  # fmt: skip
    parts = _PAGE_SPLIT.split(markdown)
    # parts = [preamble, "0", page-0 text, "1", page-1 text, ...]
    pages = list(zip(parts[1::2], parts[2::2], strict=False)) or [("0", markdown)]
    for number, text in pages:
        markdown_into(parsed, text, page=int(number) + 1)
    parsed.page_count = len(pages)
    return parsed
