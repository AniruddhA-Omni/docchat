"""Retrieval agent: hybrid search (BGE-M3 dense + BM25 sparse, RRF) -> cross-encoder rerank ->
per-document diversity cap -> neighbour expansion for very short chunks."""

from __future__ import annotations

from collections import Counter

from docchat.agents.types import AgentContext, Evidence, SubQuery, evidence_from_hit
from docchat.index.store import Hit, SearchMode

SHORT_CHUNK_TOKENS = 60
MAX_PER_DOC = 3


def probe(ctx: AgentContext, query: str, doc_ids: list[str] | None, limit: int = 20) -> list[Hit]:
    """Cheap hybrid search without reranking (used by the router)."""
    svc = ctx.services
    return svc.store.search(svc.embedder.embed_query(query), svc.sparse.encode_query(query),
                            limit=limit, doc_ids=doc_ids)  # fmt: skip


def search(ctx: AgentContext, query: str, doc_ids: list[str] | None, top_n: int | None = None,
           mode: SearchMode = "hybrid", rerank: bool = True) -> list[Hit]:  # fmt: skip
    svc, settings = ctx.services, ctx.settings
    top_n = top_n or settings.rerank_top_n
    dense = svc.embedder.embed_query(query) if mode != "sparse" else None
    sparse = svc.sparse.encode_query(query) if mode != "dense" else None
    hits = svc.store.search(dense, sparse, limit=settings.retrieve_k, doc_ids=doc_ids, mode=mode)
    reranker = svc.reranker if rerank else None
    if reranker and hits:
        scores = reranker.score(query, [h.chunk.embed_text for h in hits])
        for h, s in zip(hits, scores, strict=True):
            h.rerank = s
        hits.sort(key=lambda h: h.rerank or 0.0, reverse=True)
    return _diversify(hits, top_n, single_doc=bool(doc_ids) and len(doc_ids) == 1)


def _diversify(hits: list[Hit], top_n: int, single_doc: bool) -> list[Hit]:
    if single_doc:
        return hits[:top_n]
    per_doc: Counter[str] = Counter()
    out = []
    for h in hits:
        if per_doc[h.chunk.doc_id] >= MAX_PER_DOC:
            continue
        per_doc[h.chunk.doc_id] += 1
        out.append(h)
        if len(out) == top_n:
            break
    return out


def expand_short(ctx: AgentContext, hits: list[Hit]) -> None:
    """Append the next chunk of the same section to very short chunks (e.g. a lone table caption)."""
    for h in hits:
        c = h.chunk
        if c.token_count >= SHORT_CHUNK_TOKENS or c.kind in ("table", "table_card", "code"):
            continue
        [nxt, *_] = ctx.services.store.get_chunks(c.doc_id, [c.ordinal + 1]) or [None]
        if nxt is not None and nxt.section_path == c.section_path and nxt.kind == c.kind:
            c.text = f"{c.text}\n{nxt.text}"
            if nxt.page_end and (c.page_end or 0) < nxt.page_end:
                c.page_end = nxt.page_end
            c.bboxes = [*c.bboxes, *nxt.bboxes]


def run(ctx: AgentContext, sub: SubQuery) -> tuple[list[Evidence], dict]:
    hits = search(ctx, sub.query, sub.doc_ids)
    expand_short(ctx, hits)
    evidence = [evidence_from_hit(h, "retrieval", sub.query) for h in hits]
    trace = {
        "agent": "retrieval",
        "query": sub.query,
        "scope": sub.label or ("filtered" if sub.doc_ids else "all documents"),
        "reranker": ctx.services.reranker.name if ctx.services.reranker else "none (dense score)",
        "hits": [
            {"file": h.chunk.file_name, "loc": h.chunk.location(), "kind": str(h.chunk.kind),
             "dense_rank": h.dense_rank, "bm25_rank": h.sparse_rank,
             "rerank": round(h.rerank, 3) if h.rerank is not None else None,
             "text": h.chunk.text[:120]}
            for h in hits
        ],
    }  # fmt: skip
    return evidence, trace
