"""Render an answer: text with numbered citations, citation popovers with visual previews,
confidence badge, and the "How I answered" trace of every agent."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from docchat.agents.types import AnswerPayload, Citation
from docchat.citations.render import citation_preview
from docchat.ui.state import current_settings, get_services

_BAND = {"high": ("green", "High confidence"), "medium": ("orange", "Medium confidence"),
         "low": ("red", "Low confidence")}  # fmt: skip
_AGENT_ICON = {"retrieval": "🔎", "table": "🧮", "vision": "👁️", "code": "💻"}


def md_escape(text: str) -> str:
    """Escape `$` (Streamlit renders $...$ as LaTeX, which mangles currency)."""
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def render_answer(payload: dict) -> None:
    answer = AnswerPayload.model_validate(payload["answer"])
    text = md_escape(answer.text)
    if answer.abstained:
        st.info(text, icon="🤷")
    else:
        st.markdown(text)
    if answer.removed_sentences:
        st.caption(f"✂️ {len(answer.removed_sentences)} statement(s) were removed because the "
                   "sources did not support them.")  # fmt: skip
    if answer.citations:
        _citation_row(answer.citations)
    _meta_row(answer, payload)
    if answer.thinking:
        with st.expander("🧠 Model reasoning (deep mode)"):
            st.markdown(md_escape(answer.thinking))
    _trace(payload)


def _citation_row(citations: list[Citation]) -> None:
    cols = st.columns(min(len(citations), 4))
    for i, cite in enumerate(citations):
        icon = _AGENT_ICON.get(cite.agent, "📎")
        label = f"{icon} [{cite.n}] {cite.file_name} · {cite.location or 'source'}"
        with cols[i % len(cols)].popover(label, width="stretch"):
            _citation_detail(cite)


def _citation_detail(cite: Citation) -> None:
    st.markdown(f"**[{cite.n}] {html.escape(cite.file_name)}** — {cite.location or ''}")
    if cite.page_approx:
        st.caption("Page is approximate (DOCX files have no fixed pages).")
    if cite.quote:
        st.markdown(f"> {md_escape(cite.quote)}")
    settings = current_settings()
    rec = get_services().catalog.get_document(cite.doc_id)
    source = Path(rec.path) if rec else None
    if cite.agent == "table" and cite.sql:
        st.code(cite.sql, language="sql")
        _table_rows(cite)
        return
    if cite.agent == "code" or cite.file_type == "code":
        _code_lines(cite, source)
        return
    png = citation_preview(cite, source, settings.data_dir / "renders" / "_cite")
    if png:
        st.image(png, caption=f"{cite.file_name} {cite.location}", width="stretch")
    with st.expander("Source text", expanded=png is None):
        st.markdown(md_escape(cite.snippet))
    if cite.agent == "vision":
        st.caption("Read from the image by the vision model — verify visually above.")


def _table_rows(cite: Citation) -> None:
    catalog = get_services().catalog
    rec = catalog.get_table(cite.table_name) if cite.table_name else None
    if rec is None:
        st.markdown(md_escape(cite.snippet))
        return
    st.caption(f"Table `{rec.name}` · {rec.label} · {rec.n_rows} rows")
    df = catalog.load_frame(rec)
    if cite.row_ids and "_row" in df.columns:
        st.caption("Cited source rows:")
        st.dataframe(df[df["_row"].isin(cite.row_ids)].drop(columns=["_is_total"], errors="ignore"),
                     hide_index=True, width="stretch")  # fmt: skip
    else:
        st.markdown(md_escape(cite.snippet.split("Result", 1)[-1]))


def _code_lines(cite: Citation, source: Path | None) -> None:
    if source and source.exists() and cite.line_start:
        lines = source.read_text(encoding="utf-8", errors="replace").split("\n")
        start = max(1, cite.line_start - 2)
        end = min(len(lines), (cite.line_end or cite.line_start) + 2)
        st.caption(f"{cite.file_name}:L{cite.line_start}-{cite.line_end}")
        st.code("\n".join(lines[start - 1 : end]), language=_lang(source.suffix), line_numbers=True)
    else:
        st.markdown(cite.snippet)


def _lang(suffix: str) -> str:
    return {".py": "python", ".js": "javascript", ".ts": "typescript", ".java": "java",
            ".cs": "csharp", ".go": "go", ".rs": "rust", ".sql": "sql"}.get(suffix.lower(), "text")  # fmt: skip


def _meta_row(answer: AnswerPayload, payload: dict) -> None:
    parts = []
    if answer.confidence:
        color, label = _BAND[answer.confidence.band]
        parts.append(f":{color}-badge[{label} · {answer.confidence.score:.0%}]")
    calls = payload.get("llm_calls", [])
    if payload.get("seconds") is not None:
        parts.append(f":gray-badge[⏱ {payload['seconds']:.1f}s]")
    if calls:
        tokens = sum(c["prompt_tokens"] + c["output_tokens"] for c in calls)
        parts.append(f":gray-badge[{len(calls)} LLM calls · {tokens:,} tokens]")
    agents = payload.get("plan", {}).get("agents", [])
    if agents:
        parts.append(" ".join(f":blue-badge[{_AGENT_ICON.get(a, '')} {a}]" for a in agents))
    if parts:
        st.markdown(" ".join(parts))
    if answer.confidence and answer.confidence.notes:
        st.caption(" · ".join(answer.confidence.notes))


def _trace(payload: dict) -> None:
    trace = payload.get("trace") or []
    if not trace:
        return
    with st.expander("🔍 How I answered"):
        plan = payload.get("plan") or {}
        if plan:
            st.markdown(f"**Router** → intent `{plan.get('intent')}`, agents "
                        + ", ".join(f"`{a}`" for a in plan.get("agents", [])))  # fmt: skip
            for r in plan.get("reasons", []):
                st.caption(f"• {r}")
        answer = AnswerPayload.model_validate(payload["answer"])
        if answer.standalone_question and payload.get("question") != answer.standalone_question:
            st.caption(f"Question as understood: _{answer.standalone_question}_")
        for entry in trace:
            _trace_entry(entry)
        calls = payload.get("llm_calls", [])
        if calls:
            st.markdown("**LLM calls**")
            st.dataframe(pd.DataFrame(calls), hide_index=True, width="stretch")
        timings = [(t["node"], t.get("seconds", 0)) for t in trace]
        st.caption("Timings: " + " · ".join(f"{n} {s:.2f}s" for n, s in timings))


def _trace_entry(entry: dict) -> None:
    node = entry.get("node")
    if node == "retrieve" and entry.get("hits"):
        st.markdown(f"**🔎 Retrieval** — `{entry.get('query', '')[:80]}` ({entry.get('scope')}, "
                    f"reranker: {entry.get('reranker')})")  # fmt: skip
        st.dataframe(pd.DataFrame(entry["hits"]), hide_index=True, width="stretch")
    elif node == "table_agent":
        st.markdown(f"**🧮 Table agent** — tables {', '.join(entry.get('tables', []))} · "
                    f"method {entry.get('method', '-')}, retries {entry.get('retries', 0)}")  # fmt: skip
        for a in entry.get("attempts", []):
            if a.get("sql"):
                st.code(a["sql"], language="sql")
            status = f"❌ {a['error']}" if a.get("error") else f"✅ {a.get('rows', 0)} rows"
            st.caption(status)
    elif node == "vision_agent":
        st.markdown("**👁️ Vision agent**")
        for img in entry.get("images", []):
            st.caption(
                f"{img.get('file')} {img.get('loc', '')}: {img.get('reply') or img.get('error', '')}"
            )
    elif node == "code_agent":
        st.markdown(f"**💻 Code agent** — symbols {entry.get('symbols')} → {entry.get('found')}")
    elif node == "gather":
        st.markdown(f"**Gather** — {entry.get('evidence')} pieces of evidence, {entry.get('selected')} "
                    f"selected · best relevance {entry.get('best_relevance')} "
                    f"(abstain below {entry.get('threshold')})")  # fmt: skip
    elif node == "verify" and entry.get("verdicts") is not None:
        st.markdown(f"**✔️ Verification** — {entry.get('claims')} claims, "
                    f"{entry.get('repaired_citations', 0)} citation(s) repaired, "
                    f"{entry.get('llm_checked', 0)} LLM-checked, {entry.get('removed', 0)} removed")  # fmt: skip
        rows = [{"claim": v["sentence"][:120], "verdict": v["label"], "by": v["checked_by"],
                 "why": v["reason"]} for v in entry["verdicts"]]  # fmt: skip
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    elif node == "understand" and entry.get("rewritten"):
        st.markdown(f"**🧠 Memory** — follow-up rewritten as _{entry.get('standalone')}_")
