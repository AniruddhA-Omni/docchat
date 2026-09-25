"""'How it works' tab: architecture diagram, runtime configuration and the latest eval report."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from docchat.config import Settings
from docchat.graph.build import build_graph
from docchat.ui.state import get_services

AGENTS = [
    ("Router / Planner", "Probe retrieval + rules (aggregation words, file names, symbols, chart words); "
     "LLM planner only for multi-part questions."),
    ("Retrieval", "Hybrid search: BGE-M3 dense + BM25 sparse in Qdrant, fused with RRF, reranked by a "
     "cross-encoder, capped per document."),
    ("Table / Data", "Text-to-SQL on DuckDB over cleaned spreadsheet tables; sqlglot guard + locked "
     "sandbox; repair loop; pandas fallback."),
    ("Vision / OCR", "OCR at ingest (RapidOCR/Tesseract/Paddle) + the multimodal LLM re-reads charts and "
     "screenshots with the question."),
    ("Code", "AST symbol table, exact symbol lookup, call-site search, line-numbered evidence."),
    ("Synthesis", "One grounded answer from budgeted, document-grouped sources with [S#] citations."),
    ("Citation / Verification", "Numeric + citation checks, automatic citation repair, batched LLM claim "
     "check, removal of unsupported statements, confidence score."),
]  # fmt: skip


def _dot() -> str:
    g = build_graph().get_graph()
    lines = ["digraph G {", 'rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", '
             'fontname="Helvetica"];']  # fmt: skip
    for n in g.nodes:
        if n not in ("__start__", "__end__"):
            lines.append(f'"{n}";')
    for e in g.edges:
        style = " [style=dashed]" if e.conditional else ""
        lines.append(f'"{e.source}" -> "{e.target}"{style};')
    lines.append("}")
    return "\n".join(lines).replace('"__start__"', '"START"').replace('"__end__"', '"END"')


def render_about(settings: Settings) -> None:
    st.markdown("### Agent graph (LangGraph)")
    st.graphviz_chart(_dot(), width="stretch")
    st.markdown("### Agents")
    st.dataframe(
        pd.DataFrame(AGENTS, columns=["agent", "what it does"]), hide_index=True, width="stretch"
    )

    st.markdown("### Runtime")
    services = get_services()
    rows = {
        "Chat / vision model": settings.llm_model,
        "Embeddings": f"{settings.embed_model} ({settings.embed_backend})",
        "Sparse": "BM25 (in-house tokenizer, Qdrant IDF)",
        "Reranker": services.reranker.name if services.reranker else "none",
        "Vector store": settings.qdrant_url or f"embedded Qdrant ({settings.qdrant_path})",
        "PDF parser / OCR": f"{settings.pdf_parser} / {settings.ocr_backend}",
        "Context window": f"{settings.num_ctx} tokens",
        "Answer mode": settings.answer_mode,
    }
    st.dataframe(
        pd.DataFrame(rows.items(), columns=["setting", "value"]), hide_index=True, width="stretch"
    )

    st.markdown("### Latest evaluation")
    reports = sorted((settings.data_dir / "eval").glob("report_*.json"))
    if not reports:
        st.caption("No evaluation yet. Run `uv run docchat eval` to score retrieval, citations, "
                   "table accuracy, abstention and latency on the golden set.")  # fmt: skip
        return
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    cols = st.columns(min(len(summary), 5) or 1)
    for i, (k, v) in enumerate(summary.items()):
        cols[i % len(cols)].metric(k.replace("_", " "), f"{v:.2f}" if isinstance(v, float) else v)
    if report.get("ablation"):
        st.markdown("Retrieval ablation")
        st.dataframe(pd.DataFrame(report["ablation"]), hide_index=True, width="stretch")
    if report.get("items"):
        with st.expander("Per-question results"):
            st.dataframe(pd.DataFrame(report["items"]), hide_index=True, width="stretch")
