"""Library tab: what was ingested and how (parser, OCR, chunks, tables) — makes the layout-aware
parsing inspectable during a demo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from docchat.ui.state import get_jobs, get_services


def render_library() -> None:
    services = get_services()
    docs = services.catalog.list_documents()
    if not docs:
        st.info("No documents yet. Upload files or click **Load demo corpus** in the sidebar.")
        return
    rows = [{
        "file": d.file_name, "status": d.status, "format": d.format, "parser": d.parser,
        "ocr": d.ocr_engine or "", "pages": d.page_count, "chunks": d.n_chunks, "tables": d.n_tables,
        "size KB": round((d.size_bytes or 0) / 1024, 1), "seconds": d.elapsed_s,
        "notes": "; ".join(d.warnings)[:120] or (d.error or ""),
    } for d in docs]  # fmt: skip
    ready = [d for d in docs if d.status == "ready"]
    c = st.columns(4)
    c[0].metric("Documents", len(ready))
    c[1].metric("Chunks", sum(d.n_chunks for d in ready))
    c[2].metric("SQL tables", sum(d.n_tables for d in ready))
    c[3].metric("Pages", sum(d.page_count or 0 for d in ready))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if get_jobs().active():
        st.caption("Indexing in progress…")
    if not ready:
        return

    doc = st.selectbox("Inspect a document", ready, format_func=lambda d: d.file_name)
    tab_chunks, tab_tables = st.tabs(["Chunks", "SQL tables"])
    with tab_chunks:
        try:
            chunks = services.store.document_chunks(doc.doc_id)
        except Exception as exc:
            st.warning(f"Vector store unavailable: {exc}")
            chunks = []
        st.caption(f"{len(chunks)} chunks. Each keeps its heading path and location; tables, figures "
                   "and code symbols are never split mid-structure.")  # fmt: skip
        for ch in chunks[:200]:
            label = f"#{ch.ordinal} · {ch.kind} · {ch.location() or '-'} · {ch.token_count} tok"
            with st.expander(label):
                st.caption(ch.ctx_header)
                if ch.kind == "code":
                    st.markdown(ch.text)
                else:
                    st.text(ch.text[:3000])
                if ch.image_path:
                    st.image(ch.image_path, width=320)
    with tab_tables:
        tables = services.catalog.tables_for(doc.doc_id)
        if not tables:
            st.caption("No tables in this document.")
        for t in tables:
            with st.expander(f"`{t.name}` · {t.label} · {t.n_rows} rows"):
                st.text(t.card)
                st.dataframe(
                    services.catalog.load_frame(t).head(200), hide_index=True, width="stretch"
                )
