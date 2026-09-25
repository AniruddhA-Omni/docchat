"""Code agent: exact symbol lookup from the catalog plus lexical search for call sites.

Evidence is shown to the model with line numbers ("17 | def list_items") so explanations and
bug reports can point to exact lines, and citations render as ``file.py:L35-39``.
"""

from __future__ import annotations

from docchat.agents import retrieval
from docchat.agents.types import AgentContext, Evidence, evidence_from_hit
from docchat.schemas import Chunk

MAX_SNIPPETS = 4


def _numbered(chunk: Chunk, start: int | None = None, end: int | None = None) -> str:
    body = chunk.text
    lines = body.split("\n")
    fence_open = lines[0].startswith("```")
    code = lines[1:-1] if fence_open and lines[-1].startswith("```") else lines
    first = chunk.line_start or 1
    out = []
    for i, line in enumerate(code):
        n = first + i
        if (start is None or n >= start) and (end is None or n <= end):
            out.append(f"{n:>4} | {line}")
    return "\n".join(out)


def run(ctx: AgentContext, question: str, symbols: list[str],
        doc_ids: list[str] | None) -> tuple[list[Evidence], dict]:  # fmt: skip
    svc = ctx.services
    trace: dict = {"agent": "code", "symbols": symbols, "found": []}
    evidence: list[Evidence] = []
    seen: set[str] = set()

    for sym in svc.catalog.find_symbols(symbols, doc_ids)[:MAX_SNIPPETS]:
        chunks = svc.store.document_chunks(sym["doc_id"], kinds=["code"])
        for c in chunks:
            if c.line_start is None or c.line_end is None:
                continue
            if c.line_end < sym["line_start"] or c.line_start > sym["line_end"]:
                continue
            code = _numbered(c, sym["line_start"], sym["line_end"])
            key = f"{c.chunk_id}:{sym['line_start']}"
            if not code or key in seen:
                continue
            seen.add(key)
            lang = c.language or ""
            evidence.append(Evidence(
                id=f"code:{key}", agent="code", kind="code", score=0.95,
                content=f"{sym['kind']} {sym['symbol']} in {c.file_name} "
                        f"(lines {sym['line_start']}-{sym['line_end']}):\n```{lang}\n{code}\n```",
                doc_id=c.doc_id, file_name=c.file_name, file_type="code",
                location=f"L{sym['line_start']}-{sym['line_end']}",
                line_start=sym["line_start"], line_end=sym["line_end"], chunk_kind="code",
            ))  # fmt: skip
            trace["found"].append(f"{sym['symbol']} @ {c.file_name}:L{sym['line_start']}")

    # Call sites / related code: lexical search restricted to code chunks.
    query = " ".join(symbols) if symbols else question
    for h in retrieval.search(ctx, query, doc_ids, top_n=6, mode="sparse" if symbols else "hybrid"):
        if h.chunk.kind != "code" or len(evidence) >= MAX_SNIPPETS + 2:
            continue
        if any(e.doc_id == h.chunk.doc_id and e.line_start and h.chunk.line_start
               and e.line_start >= h.chunk.line_start and e.line_end <= (h.chunk.line_end or 0)
               for e in evidence):  # fmt: skip
            continue  # already covered by a symbol snippet
        ev = evidence_from_hit(h, "code", query)
        ev.content = (
            f"Code from {h.chunk.file_name} ({h.chunk.location()}):\n```\n{_numbered(h.chunk)}\n```"
        )
        evidence.append(ev)
    return evidence, trace
