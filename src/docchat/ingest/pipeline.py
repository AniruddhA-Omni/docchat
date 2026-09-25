"""Ingestion pipeline: file -> parse -> tables/symbols/captions -> chunks -> embeddings -> index.

Documents are identified by content hash, so re-uploading the same file is a no-op unless the
parser, chunker or embedding model changed (tracked in ``index_key``).
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from docchat.config import Settings
from docchat.ingest.chunking import CHUNKER_VERSION, chunk_document
from docchat.ingest.context import ParseContext
from docchat.ingest.detect import Format, detect_format
from docchat.ingest.registry import parse_file
from docchat.ingest.safety import safe_filename
from docchat.ingest.tables import table_card
from docchat.llm import LLMError
from docchat.schemas import Element, Kind, ParsedDocument, file_digest
from docchat.services import Services

log = logging.getLogger(__name__)
Progress = Callable[[str], None]
EMBED_BATCH = 32
MIN_CAPTION_SIDE = 96

CAPTION_PROMPT = (
    "Describe this image so it can be found by search and used to answer questions. "
    "If it is a chart: give its title, axes, units and every data label or value you can read. "
    "If it is a screenshot or dashboard: list every label with its value. "
    "If it is a scanned document: summarise what document it is and its key fields. "
    "Otherwise describe what it shows. Be factual and concise; do not guess unreadable values."
)


@dataclass
class IngestResult:
    file_name: str
    status: str  # ready | skipped | error
    doc_id: str = ""
    n_chunks: int = 0
    n_tables: int = 0
    seconds: float = 0.0
    error: str = ""
    warnings: list[str] = field(default_factory=list)


def index_key(settings: Settings, fmt: Format, embed_name: str) -> str:
    parser = settings.pdf_parser if fmt == Format.PDF else fmt.value
    return f"chunker{CHUNKER_VERSION}|{parser}|{settings.ocr_backend}|{embed_name}"


def _store_copy(src: Path, settings: Settings, doc_id: str) -> Path:
    """Keep a stable copy under data/uploads (citations render pages from it)."""
    uploads = settings.uploads_dir.resolve()
    if src.resolve().parent == uploads:
        return src
    dest = uploads / safe_filename(src.name)
    if dest.exists() and file_digest(dest)[:16] != doc_id:
        dest = uploads / f"{dest.stem}_{doc_id[:6]}{dest.suffix}"
    if not dest.exists():
        shutil.copy2(src, dest)
    return dest


def ingest_file(services: Services, settings: Settings, src: Path, progress: Progress | None = None,
                force: bool = False) -> IngestResult:  # fmt: skip
    report = progress or (lambda _msg: None)
    started = time.perf_counter()
    fmt = detect_format(src)
    if fmt is None:
        return IngestResult(src.name, "error", error=f"unsupported file type {src.suffix}")
    doc_id = file_digest(src)[:16]
    key = index_key(settings, fmt, services.embedder.name)
    catalog = services.catalog
    existing = catalog.get_document(doc_id)
    if existing and existing.status == "ready" and existing.index_key == key and not force:
        return IngestResult(existing.file_name, "skipped", doc_id=doc_id,
                            n_chunks=existing.n_chunks, n_tables=existing.n_tables)  # fmt: skip

    with services.ingest_lock:
        path = _store_copy(src, settings, doc_id)
        if existing:
            _remove_index_data(services, settings, doc_id)
        from docchat.index.catalog import DocRecord

        record = DocRecord(doc_id=doc_id, sha256=file_digest(path), file_name=path.name,
                           path=str(path), format=fmt.value, status="processing",
                           size_bytes=path.stat().st_size, index_key=key,
                           added_at=time.time())  # fmt: skip
        catalog.upsert_document(record)
        try:
            report(f"Parsing {path.name}")
            ctx = ParseContext(settings, doc_id, settings.data_dir / "renders" / doc_id, report)
            parsed = parse_file(path, ctx)
            _register_tables(services, parsed)
            if parsed.symbols:
                catalog.add_symbols(doc_id, parsed.symbols)
            if settings.caption_images:
                caption_figures(services, settings, parsed, report)
            chunks = chunk_document(parsed, settings)
            if not chunks:
                raise ValueError("no text, tables or images could be extracted")
            for i in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[i : i + EMBED_BATCH]
                report(
                    f"Embedding {path.name}: {min(i + EMBED_BATCH, len(chunks))}/{len(chunks)} chunks"
                )
                texts = [c.embed_text for c in batch]
                dense = services.embedder.embed_documents(texts)
                sparse = [services.sparse.encode_document(t) for t in texts]
                services.store.upsert(batch, dense, sparse)
        except Exception as exc:
            log.exception("ingestion failed for %s", path.name)
            _remove_index_data(services, settings, doc_id, keep_record=True)
            record.status, record.error = "error", str(exc)[:500]
            record.elapsed_s = round(time.perf_counter() - started, 2)
            catalog.upsert_document(record)
            return IngestResult(path.name, "error", doc_id=doc_id, error=record.error)

        record.status = "ready"
        record.parser, record.ocr_engine = parsed.parser, parsed.ocr_engine
        record.page_count, record.n_chunks, record.n_tables = (
            parsed.page_count,
            len(chunks),
            len(parsed.tables),
        )
        record.warnings = parsed.warnings
        record.elapsed_s = round(time.perf_counter() - started, 2)
        catalog.upsert_document(record)
        report(f"Indexed {path.name}: {len(chunks)} chunks, {len(parsed.tables)} tables")
        return IngestResult(path.name, "ready", doc_id, len(chunks), len(parsed.tables),
                            record.elapsed_s, warnings=parsed.warnings)  # fmt: skip


def ingest_many(services: Services, settings: Settings, paths: Iterable[Path],
                progress: Progress | None = None, force: bool = False) -> list[IngestResult]:  # fmt: skip
    return [ingest_file(services, settings, p, progress, force) for p in paths]


def _register_tables(services: Services, parsed: ParsedDocument) -> None:
    """Store each table as Parquet for SQL and link it to its chunk (or create a table card)."""
    stem = Path(parsed.file_name).stem
    for idx, table in enumerate(parsed.tables):
        base = f"{stem}_{table.sheet}" if table.sheet else (
            f"{stem}_p{table.page}" if table.page and parsed.format != "tabular" else stem)  # fmt: skip
        table.name = services.catalog.unique_table_name(base)
        card = table_card(
            table, parsed.file_name, sample_rows=len(table.df) if len(table.df) <= 12 else 3
        )
        services.catalog.add_table(parsed.doc_id, table, card)
        linked = next((e for e in parsed.elements if e.meta.get("table_idx") == idx), None)
        if linked is not None:
            linked.meta["table_name"] = table.name
        else:  # spreadsheets: the card is the searchable representation of the table
            parsed.elements.append(Element(kind=Kind.TABLE_CARD, text=card, page=None,
                                           meta={"table_name": table.name, "sheet": table.sheet}))  # fmt: skip


def caption_figures(services: Services, settings: Settings, parsed: ParsedDocument,
                    report: Progress) -> None:  # fmt: skip
    figures = [e for e in parsed.elements if e.kind == Kind.FIGURE and e.image_path]
    if not figures:
        return
    llm = services.vision_llm(settings)
    if not llm.supports_vision:
        parsed.warnings.append(f"{llm.model} has no vision capability; figures not described")
        return
    from PIL import Image

    for k, el in enumerate(figures[: settings.max_captions_per_doc], 1):
        with Image.open(el.image_path) as im:
            if min(im.size) < MIN_CAPTION_SIDE:
                continue
        report(f"Describing image {k}/{min(len(figures), settings.max_captions_per_doc)} "
               f"in {parsed.file_name}")  # fmt: skip
        hint = el.meta.get("ocr_text", "")
        prompt = CAPTION_PROMPT + (f"\nOCR text found in the image (may contain errors): {hint[:1500]}"
                                   if hint else "")  # fmt: skip
        try:
            res = llm.chat([{"role": "user", "content": prompt}], images=[el.image_path],
                           tag="caption", think=False, temperature=0.0, num_predict=400)  # fmt: skip
        except LLMError as exc:
            parsed.warnings.append(f"image description failed: {exc}")
            return
        if res.content:
            el.text = f"{el.text}\nDescription ({llm.model}): {res.content}"
            el.meta["caption_model"] = llm.model
    if len(figures) > settings.max_captions_per_doc:
        parsed.warnings.append(f"only the first {settings.max_captions_per_doc} of {len(figures)} "
                               "images were described")  # fmt: skip


def _remove_index_data(services: Services, settings: Settings, doc_id: str,
                       keep_record: bool = False) -> None:  # fmt: skip
    try:
        services.store.delete_document(doc_id)
    except Exception as exc:  # store may not exist yet
        log.debug("store delete failed: %s", exc)
    services.catalog.delete_document(doc_id, keep_record=keep_record)
    shutil.rmtree(settings.data_dir / "renders" / doc_id, ignore_errors=True)


def delete_document(services: Services, settings: Settings, doc_id: str) -> None:
    rec = services.catalog.get_document(doc_id)
    _remove_index_data(services, settings, doc_id)
    if rec:
        path = Path(rec.path)
        if path.parent.resolve() == settings.uploads_dir.resolve():
            path.unlink(missing_ok=True)
