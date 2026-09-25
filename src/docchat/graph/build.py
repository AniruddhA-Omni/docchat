"""The multi-agent graph.

prepare -> understand (follow-up rewrite) -> plan (router)
    -> Send: retrieve x N | table_agent | vision_agent | code_agent   (parallel)
    -> gather (fan-in, evidence budget, "I don't know" gate)
    -> synthesize (streamed, [S#] citations) -> verify (rules + LLM, repair, confidence)
    -> finalize (renumber citations, update memory)
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Overwrite, Send

from docchat.agents import code as code_agent
from docchat.agents import memory, retrieval, synthesis, verification
from docchat.agents import table as table_agent
from docchat.agents import vision as vision_agent
from docchat.agents.confidence import score as confidence_score
from docchat.agents.router import CHITCHAT_RE, plan_question
from docchat.agents.types import AgentContext, AnswerPayload, Evidence, Plan, SubQuery
from docchat.citations.parse import renumber
from docchat.graph.state import GraphState
from docchat.llm import LLMError

NO_DOCS = "No documents are indexed yet (or none are selected). Upload files in the sidebar first."
IDK = "I don't know — I couldn't find this in your documents."
CHITCHAT = "Hi! Ask me anything about the files in your library — I'll answer with citations."


def _emit(event: dict) -> None:
    with contextlib.suppress(Exception):  # no stream writer outside a streaming run
        get_stream_writer()(event)


def node(name: str, status: str | None = None) -> Callable:
    """Wrap a node: emit a status line, time it, and add a trace entry."""

    def deco(fn):
        # No functools.wraps: LangGraph inspects this signature to inject ``runtime``.
        def wrapper(state, runtime: Runtime[AgentContext]):
            if status:
                _emit({"type": "status", "text": status})
            start = time.perf_counter()
            out = fn(state, runtime.context) or {}
            out = dict(out)
            entry = {"node": name, "seconds": round(time.perf_counter() - start, 2)}
            extra = out.pop("_trace", None)
            if extra:
                entry.update(extra)
            reset = out.get("trace")
            out["trace"] = (
                Overwrite([*reset.value, entry]) if isinstance(reset, Overwrite) else [entry]
            )
            return out

        wrapper.__name__ = fn.__name__
        return wrapper

    return deco


# ------------------------------------------------------------------------------------------------
@node("prepare")
def prepare(state: GraphState, ctx: AgentContext) -> dict:
    ready = [d.doc_id for d in ctx.services.catalog.ready_documents()]
    scope = state.get("scope_doc_ids")
    doc_ids = [d for d in ready if scope is None or d in scope]
    return {
        "evidence": Overwrite([]),
        "trace": Overwrite([]),
        "sources": [],
        "draft": "",
        "thinking": "",
        "answer": {},
        "abstain": "" if doc_ids else "no_documents",
        "doc_ids": doc_ids,
        "standalone": state["question"],
        "followup": False,
        "plan": {},
    }


@node("understand", "Understanding the question")
def understand(state: GraphState, ctx: AgentContext) -> dict:
    history, question = state.get("history", []), state["question"]
    if CHITCHAT_RE.match(question) or not memory.is_followup(question, history):
        return {"standalone": question, "followup": False}
    standalone = memory.rewrite(ctx, question, history, state.get("summary", ""))
    if standalone != question:
        _emit({"type": "status", "text": f"Follow-up understood as: {standalone}"})
    return {"standalone": standalone, "followup": True,
            "_trace": {"rewritten": standalone != question, "standalone": standalone}}  # fmt: skip


@node("plan", "Planning which agents to use")
def plan(state: GraphState, ctx: AgentContext) -> dict:
    question, doc_ids = state["standalone"], state["doc_ids"]
    docs = [d for d in ctx.services.catalog.ready_documents() if d.doc_id in doc_ids]
    probe = retrieval.probe(ctx, question, doc_ids)
    p = plan_question(ctx, question, docs, probe)
    active = [d for d in state.get("active_doc_ids", []) if d in doc_ids]
    if state.get("followup") and active and p.intent != "chitchat" and not p.doc_filter:
        p.sub_queries.append(
            SubQuery(query=question, doc_ids=active, label="documents from the previous answer")
        )
        p.reasons.append("follow-up: also searching the documents used in the previous answer")
    return {"plan": p.model_dump(), "_trace": {"plan": p.model_dump(exclude={"sub_queries"}),
                                               "sub_queries": [s.query for s in p.sub_queries]}}  # fmt: skip


def dispatch(state: GraphState) -> list[Send] | str:
    if state.get("abstain"):
        return "finalize"
    p = Plan.model_validate(state["plan"])
    if p.intent == "chitchat":
        return "finalize"
    sends = [Send("retrieve", {"sub": s.model_dump()}) for s in p.sub_queries]
    common = {"question": state["standalone"], "doc_ids": p.doc_filter or state["doc_ids"]}
    if "table" in p.agents:
        sends.append(Send("table_agent", {**common, "tables": p.table_names,
                                          "last_sql": state.get("last_sql")}))  # fmt: skip
    if "vision" in p.agents:
        sends.append(Send("vision_agent", {**common, "figures": p.figure_ids}))
    if "code" in p.agents:
        sends.append(Send("code_agent", {**common, "symbols": p.symbols}))
    return sends or "gather"


# --- worker agents (receive Send payloads, not the full state) -----------------------------------
def _worker(name: str, status: str, fn):
    @node(name, status)
    def run(payload: dict, ctx: AgentContext) -> dict:
        try:
            evidence, trace = fn(payload, ctx)
        except LLMError as exc:
            evidence, trace = [], {"error": str(exc)[:300]}
        return {"evidence": [e.model_dump() for e in evidence], "_trace": trace}

    return run


retrieve = _worker("retrieve", "Searching documents (hybrid BM25 + vector, reranked)",
                   lambda p, ctx: retrieval.run(ctx, SubQuery.model_validate(p["sub"])))  # fmt: skip
table_node = _worker("table_agent", "Querying spreadsheets with SQL",
                     lambda p, ctx: table_agent.run(ctx, p["question"], p["tables"], p.get("last_sql")))  # fmt: skip
vision_node = _worker("vision_agent", "Looking at images and charts",
                      lambda p, ctx: vision_agent.run(ctx, p["question"], p["figures"]))  # fmt: skip
code_node = _worker("code_agent", "Reading source code",
                    lambda p, ctx: code_agent.run(ctx, p["question"], p["symbols"], p["doc_ids"]))  # fmt: skip


# ------------------------------------------------------------------------------------------------
@node("gather")
def gather(state: GraphState, ctx: AgentContext) -> dict:
    evidence = [Evidence.model_validate(e) for e in state.get("evidence", [])]
    sources = synthesis.select_evidence(evidence, ctx.settings.evidence_budget_tokens)
    specialist = [e for e in sources if e.agent != "retrieval" and e.score >= 0.5]
    retrieved = [e for e in sources if e.agent == "retrieval"]
    best = max((e.score for e in retrieved), default=0.0)
    threshold = (ctx.settings.abstain_rerank_threshold if ctx.services.reranker
                 else ctx.settings.abstain_dense_threshold)  # fmt: skip
    abstain = "" if (specialist or best >= threshold) else "low_relevance"
    return {"sources": [e.model_dump() for e in sources], "abstain": abstain,
            "_trace": {"evidence": len(evidence), "selected": len(sources),
                       "best_relevance": round(best, 3), "threshold": threshold}}  # fmt: skip


def after_gather(state: GraphState) -> str:
    return "finalize" if state.get("abstain") else "synthesize"


@node("synthesize", "Writing the answer")
def synthesize(state: GraphState, ctx: AgentContext) -> dict:
    sources = [Evidence.model_validate(e) for e in state["sources"]]
    p = Plan.model_validate(state["plan"])
    history = memory.history_text(state.get("history", []), state.get("summary", ""), turns=2,
                                  answer_chars=250) if state.get("followup") else ""  # fmt: skip
    messages = synthesis.build_messages(state["standalone"], sources, p.intent, history)
    think = state.get("mode") == "deep"
    _emit({"type": "answer_start"})
    res = synthesis.write_answer(
        ctx, messages, think,
        on_token=lambda t: _emit({"type": "token", "text": t}),
        on_thinking=lambda t: _emit({"type": "thinking", "text": t}),
    )  # fmt: skip
    if synthesis.is_not_found(res.content):
        return {"draft": "", "abstain": "not_found", "thinking": res.thinking}
    return {"draft": res.content, "thinking": res.thinking,
            "_trace": {"prompt_tokens": res.prompt_tokens, "output_tokens": res.output_tokens,
                       "sources": len(sources)}}  # fmt: skip


def after_synthesize(state: GraphState) -> str:
    return "finalize" if state.get("abstain") else "verify"


@node("verify", "Verifying every claim against its sources")
def verify(state: GraphState, ctx: AgentContext) -> dict:
    sources = [Evidence.model_validate(e) for e in state["sources"]]
    mode = state.get("mode", "balanced")
    draft, verdicts, stats = verification.verify(
        ctx, state["draft"], sources, use_llm=mode != "fast"
    )
    removed: list[str] = []
    unsupported = [v for v in verdicts if v.label == "unsupported"]
    if unsupported and mode == "deep":
        # Deep mode: one LLM revision pass with the flagged claims, then re-check with the rules.
        _emit({"type": "status", "text": f"Revising {len(unsupported)} unverified statement(s)"})
        revised = verification.revise(ctx, state["standalone"], draft, unsupported, sources)
        if revised:
            draft, verdicts, again = verification.verify(ctx, revised, sources, use_llm=False)
            stats = {**stats, "revised": True, "repaired_citations":
                     stats["repaired_citations"] + again["repaired_citations"]}  # fmt: skip
            unsupported = [v for v in verdicts if v.label == "unsupported"]
    if unsupported:
        supported = [v for v in verdicts if v.label in ("supported", "partial")]
        if not supported:
            return {"abstain": "unverified", "draft": draft,
                    "_trace": {**stats, "verdicts": [v.model_dump() for v in verdicts]}}  # fmt: skip
        draft, removed = verification.remove_unsupported(draft, verdicts)
        verdicts = [v for v in verdicts if v.label != "unsupported"]
        _emit({"type": "status", "text": f"Removed {len(removed)} unverified statement(s)"})
    text, citations = renumber(draft, sources)
    conf = confidence_score(verdicts, citations)
    answer = AnswerPayload(text=text, citations=citations, verdicts=verdicts, confidence=conf,
                           standalone_question=state["standalone"], removed_sentences=removed,
                           thinking=state.get("thinking", ""))  # fmt: skip
    return {"answer": answer.model_dump(), "draft": draft,
            "_trace": {**stats, "removed": len(removed), "confidence": conf.score,
                       "verdicts": [v.model_dump() for v in verdicts]}}  # fmt: skip


@node("finalize")
def finalize(state: GraphState, ctx: AgentContext) -> dict:
    reason = state.get("abstain", "")
    p = Plan.model_validate(state["plan"]) if state.get("plan") else None
    if state.get("answer") and not reason:
        answer = AnswerPayload.model_validate(state["answer"])
    else:
        answer = _abstention(state, reason, p)
    history = state.get("history", [])
    turn = {"q": state["question"], "standalone": state.get("standalone", state["question"]),
            "a": answer.text, "docs": list(dict.fromkeys(c.doc_id for c in answer.citations))}  # fmt: skip
    out: dict = {"answer": answer.model_dump(), "history": [turn]}
    if turn["docs"]:
        out["active_doc_ids"] = turn["docs"]
    sql = next((c for c in answer.citations if c.sql and c.agent == "table"), None)
    if sql:
        out["last_sql"] = {"sql": sql.sql, "tables": [sql.table_name]}
    if len(history) + 1 > memory.SUMMARIZE_AFTER and (len(history) + 1) % 4 == 0:
        out["summary"] = memory.summarize(ctx, [*history, turn], state.get("summary", ""))
    return out


def _abstention(state: GraphState, reason: str, p: Plan | None) -> AnswerPayload:
    if p is not None and p.intent == "chitchat":
        return AnswerPayload(text=CHITCHAT, standalone_question=state.get("standalone", ""))
    if reason == "no_documents":
        return AnswerPayload(text=NO_DOCS, abstained=True, abstain_reason=reason)
    nearest = []
    for e in state.get("sources", [])[:3] or state.get("evidence", [])[:3]:
        nearest.append(f"{e['file_name']} {e.get('location', '')}".strip())
    detail = {
        "low_relevance": "None of the indexed passages is relevant enough to answer confidently.",
        "not_found": "The most relevant passages do not contain the answer.",
        "unverified": "I drafted an answer, but none of its statements could be verified against the sources.",
    }.get(reason, "")
    text = f"{IDK} {detail}".strip()
    if nearest:
        text += "\n\nClosest sources I checked: " + "; ".join(dict.fromkeys(nearest)) + "."
    return AnswerPayload(text=text, abstained=True, abstain_reason=reason, nearest_sources=nearest,
                         standalone_question=state.get("standalone", ""))  # fmt: skip


# ------------------------------------------------------------------------------------------------
def build_graph(checkpointer=None):
    g = StateGraph(GraphState, context_schema=AgentContext)
    g.add_node("prepare", prepare)
    g.add_node("understand", understand)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("table_agent", table_node)
    g.add_node("vision_agent", vision_node)
    g.add_node("code_agent", code_node)
    g.add_node("gather", gather, defer=True)
    g.add_node("synthesize", synthesize)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)

    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", lambda s: "finalize" if s.get("abstain") else "understand",
                            ["understand", "finalize"])  # fmt: skip
    g.add_edge("understand", "plan")
    g.add_conditional_edges("plan", dispatch, ["retrieve", "table_agent", "vision_agent",
                                               "code_agent", "gather", "finalize"])  # fmt: skip
    for worker in ("retrieve", "table_agent", "vision_agent", "code_agent"):
        g.add_edge(worker, "gather")
    g.add_conditional_edges("gather", after_gather, ["synthesize", "finalize"])
    g.add_conditional_edges("synthesize", after_synthesize, ["verify", "finalize"])
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
