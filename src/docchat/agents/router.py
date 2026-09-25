"""Router / Planner agent.

Routing is rule-first: a cheap probe retrieval (no reranking) plus signals from the question
(aggregation words, chart/code vocabulary, file names, known symbols and column names) decide
which specialist agents run. The LLM planner is only consulted for multi-part questions, which
keeps a 4B model's latency budget for the answer itself.
"""

from __future__ import annotations

import re

from pydantic import BaseModel
from rapidfuzz import fuzz

from docchat.agents.types import AgentContext, Plan, SubQuery
from docchat.index.catalog import DocRecord, TableRecord
from docchat.index.sparse import tokenize
from docchat.index.store import Hit
from docchat.llm import LLMError
from docchat.llm.prompts import PLANNER_SYSTEM

AGG_RE = re.compile(
    r"\b(how many|how much|total|sum|average|avg|mean|median|count|number of|max(imum)?|min(imum)?|"
    r"highest|lowest|largest|smallest|most|least|top \d*|bottom|rank|per (region|product|month|"
    r"quarter|year|customer|category)|by (region|product|month|quarter|year|customer|category)|"
    r"group|breakdown|distribution|percentage|share|growth|trend|compare .* (sales|revenue)|"
    r"missed|exceed|above|below|greater than|less than|between|sorted|list all|which .* (had|have))\b",
    re.I,
)
VISION_RE = re.compile(r"\b(chart|graph|plot|figure|fig\.?|image|picture|photo|screenshot|"
                       r"dashboard|diagram|visual|scan(ned)?|logo|drawing|slide image)\b", re.I)  # fmt: skip
CODE_RE = re.compile(r"\b(code|function|method|class|bug|error|exception|implement(ation|ed)?|"
                     r"variable|return(s)?|parameter|argument|module|api|refactor|pagination|"
                     r"algorithm|loop|call(s|ed)?)\b|\w+\(\)|`[^`]+`", re.I)  # fmt: skip
COMPARE_RE = re.compile(r"\b(compare|comparison|difference|differ|vs\.?|versus|across|both|each "
                        r"(document|file)|all (the )?(documents|files)|conflict|disagree|"
                        r"consistent|contradict)\b", re.I)  # fmt: skip
SUMMARY_RE = re.compile(r"\b(summari[sz]e|summary|overview|key (points|takeaways)|tl;?dr|"
                        r"what is (this|the) (document|file) about|main points)\b", re.I)  # fmt: skip
CHITCHAT_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|good (morning|evening))\W*$", re.I
)


class LLMPlan(BaseModel):
    sub_questions: list[str] = []
    tools: list[str] = []


def mentioned_documents(question: str, docs: list[DocRecord]) -> list[DocRecord]:
    """Documents whose file name (or stem) is referenced in the question."""
    q = question.lower()
    out = []
    for d in docs:
        stem = re.sub(r"[_\-.]+", " ", d.file_name.rsplit(".", 1)[0].lower()).strip()
        if d.file_name.lower() in q or (len(stem) > 4 and fuzz.partial_ratio(stem, q) >= 90):
            out.append(d)
    return out


def match_tables(question: str, tables: list[TableRecord]) -> list[tuple[float, TableRecord]]:
    """Score tables by overlap between question words and table/sheet/column names."""
    q_tokens = set(tokenize(question))
    scored = []
    for t in tables:
        names = [t.name, t.sheet or "", t.title or "", t.label, *t.columns, *t.columns.values()]
        t_tokens = set(tokenize(" ".join(names)))
        overlap = len(q_tokens & t_tokens)
        if overlap:
            scored.append((overlap / (len(q_tokens) or 1), t))
    return sorted(scored, key=lambda x: x[0], reverse=True)


def is_multi_part(question: str) -> bool:
    words = question.split()
    return len(words) > 14 and (question.count("?") > 1 or bool(re.search(
        r"\b(and also|as well as|and then|and what|and how|and which|and who|and when)\b",
        question, re.I)))  # fmt: skip


def plan_question(ctx: AgentContext, question: str, docs: list[DocRecord],
                  probe: list[Hit]) -> Plan:  # fmt: skip
    plan = Plan(agents=[], reasons=[])
    if CHITCHAT_RE.match(question):
        return Plan(intent="chitchat", agents=[], reasons=["greeting / small talk"])

    doc_ids = [d.doc_id for d in docs]
    named = mentioned_documents(question, docs)
    if named:
        plan.doc_filter = [d.doc_id for d in named]
        plan.reasons.append("question names " + ", ".join(d.file_name for d in named))
    scope_ids = plan.doc_filter or doc_ids
    top = probe[:8]

    # --- table agent ----------------------------------------------------------------------------
    tables = [t for t in ctx.services.catalog.tables_for() if t.doc_id in scope_ids]
    hit_tables = [h.chunk.table_name for h in probe[:5] if h.chunk.table_name]
    name_matches = [t.name for score, t in match_tables(question, tables) if score >= 0.15]
    # A spreadsheet is the best match for the question even without aggregation words
    # (e.g. "what is the APAC target?"): top hit, found by both dense and keyword search.
    spreadsheet_hit = (
        bool(probe)
        and probe[0].chunk.kind == "table_card"
        and (probe[0].sparse_rank is not None and probe[0].sparse_rank <= 2)
    )
    if tables and (AGG_RE.search(question) or spreadsheet_hit):
        chosen = list(dict.fromkeys(hit_tables + name_matches))
        if not chosen and len(tables) <= 3:
            chosen = [t.name for t in tables]
        if chosen:
            plan.agents.append("table")
            plan.table_names = chosen[:3]
            plan.reasons.append(
                "aggregation / lookup over structured data" if AGG_RE.search(question)
                else "top search hits are spreadsheet tables")  # fmt: skip

    # --- vision agent ---------------------------------------------------------------------------
    figure_hits = [h for h in top if h.chunk.kind == "figure" and h.chunk.image_path]
    wants_visual = bool(VISION_RE.search(question))
    if figure_hits and (wants_visual or figure_hits[0] in probe[:2]):
        limit = 2 if ctx.settings.llm_parallel > 1 else 1
        plan.agents.append("vision")
        plan.figure_ids = [h.chunk.chunk_id for h in figure_hits[:limit]]
        plan.reasons.append("question refers to a chart / image" if wants_visual
                            else "an image is among the top search hits")  # fmt: skip

    # --- code agent -----------------------------------------------------------------------------
    code_docs = [d for d in docs if d.format == "code" and d.doc_id in scope_ids]
    if code_docs:
        known = ctx.services.catalog.all_symbol_names()
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", question))
        symbols = sorted(w for w in words if w in known and len(w) > 2)
        code_hit = any(h.chunk.kind == "code" for h in probe[:3])
        if symbols or (CODE_RE.search(question) and (code_hit or named)):
            plan.agents.append("code")
            plan.symbols = symbols
            plan.reasons.append(f"code symbols {', '.join(symbols)}" if symbols
                                else "question is about source code")  # fmt: skip

    # --- intent + retrieval sub-queries --------------------------------------------------------
    if SUMMARY_RE.search(question):
        plan.intent = "summary"
    elif COMPARE_RE.search(question) or len(named) > 1:
        plan.intent = "compare"
    elif "table" in plan.agents:
        plan.intent = "table"
    elif "code" in plan.agents:
        plan.intent = "code"
    elif "vision" in plan.agents:
        plan.intent = "vision"

    plan.agents.insert(0, "retrieval")
    if plan.intent in ("compare", "summary") and len(named) > 1:
        plan.sub_queries = [SubQuery(query=question, doc_ids=[d.doc_id], label=d.file_name)
                            for d in named[:4]]  # fmt: skip
        plan.reasons.append("one search per named document")
    elif plan.intent == "compare":
        # Cross-document question without explicit files: search each of the top documents.
        top_docs = list(dict.fromkeys(h.chunk.doc_id for h in probe[:10]))[:3]
        plan.sub_queries = [SubQuery(query=question, doc_ids=plan.doc_filter)]
        plan.sub_queries += [SubQuery(query=question, doc_ids=[d], label=d) for d in top_docs]
        plan.reasons.append("cross-document question: per-document searches")
    else:
        plan.sub_queries = [SubQuery(query=question, doc_ids=plan.doc_filter)]

    if is_multi_part(question):
        _llm_decompose(ctx, question, plan)
    return plan


def _llm_decompose(ctx: AgentContext, question: str, plan: Plan) -> None:
    llm = ctx.services.chat_llm(ctx.settings)
    try:
        out = llm.json([{"role": "user", "content": question}], LLMPlan, system=PLANNER_SYSTEM,
                       tag="planner", num_predict=200)  # fmt: skip
    except LLMError:
        return
    plan.used_llm = True
    subs = [s.strip() for s in out.sub_questions if s.strip()][:3]
    if len(subs) > 1:
        plan.sub_queries = [SubQuery(query=s, doc_ids=plan.doc_filter) for s in subs]
        plan.reasons.append(f"LLM planner split the question into {len(subs)} parts")
    tool_map = {"table": "table", "vision": "vision", "code": "code"}
    for tool in out.tools:
        agent = tool_map.get(tool)
        if agent and agent not in plan.agents:
            if agent == "table" and not plan.table_names:
                continue  # nothing to query
            if agent == "vision" and not plan.figure_ids:
                continue
            plan.agents.append(agent)
