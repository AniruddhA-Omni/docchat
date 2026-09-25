"""End-to-end agent graph tests with a scripted LLM and the hashing embedder (no Ollama)."""

from __future__ import annotations

import pandas as pd
import pytest

from docchat.agents.router import mentioned_documents, plan_question
from docchat.agents.types import AgentContext
from docchat.graph.run import ask, mermaid, stream_turn
from docchat.ingest.pipeline import ingest_file
from docchat.services import Services
from tests.fakes import FakeLLM

FILES = ["engineering_handbook.md", "meeting_notes_2025-11-03.txt", "inventory_service.py",
         "sales_2025.xlsx", "support_tickets.csv", "board_update_q4.pptx"]  # fmt: skip


@pytest.fixture
def graph_env(settings, corpus):
    settings = settings.model_copy(update={"abstain_dense_threshold": 0.1})
    fake = FakeLLM()
    svc = Services(settings, llm=fake)
    for name in FILES:
        assert ingest_file(svc, settings, corpus / name).status == "ready"
    yield svc, settings, fake
    svc.close()


def test_lookup_answer_is_cited_and_verified(graph_env):
    svc, settings, _ = graph_env
    done = ask(svc, settings, "t1", "When are production deploys frozen?")
    a = done["answer"]
    assert not a.abstained
    assert "December 20" in a.text and "[1]" in a.text
    assert a.citations[0].file_name == "engineering_handbook.md"
    assert a.citations[0].location.startswith("L")
    assert a.confidence and a.confidence.band == "high"
    assert [t["node"] for t in done["trace"]][:3] == ["prepare", "understand", "plan"]


def test_table_question_uses_sql_with_exact_numbers(graph_env, corpus):
    svc, settings, fake = graph_env
    done = ask(svc, settings, "t1", "What is the total revenue for Europe in the Orders sheet?")
    orders = pd.read_excel(corpus / "sales_2025.xlsx", sheet_name="Orders")
    expected = int(orders.loc[orders.region == "Europe", "revenue"].sum())
    assert str(expected) in done["answer"].text
    cite = done["answer"].citations[0]
    assert cite.agent == "table" and cite.sheet == "Orders" and "SUM(revenue)" in cite.sql
    assert "table" in done["plan"]["agents"]
    assert "text-to-sql" in [c["tag"] for c in done["llm_calls"]]


def test_followup_is_rewritten_using_memory(graph_env, corpus):
    svc, settings, _ = graph_env
    ask(svc, settings, "t2", "How many support tickets are in the log?")
    done = ask(svc, settings, "t2", "How many of them are critical?")
    tickets = pd.read_csv(corpus / "support_tickets.csv")
    assert str(int((tickets.severity == "critical").sum())) in done["answer"].text
    assert done["answer"].standalone_question == "How many of the support tickets are critical?"
    state = svc.checkpointer.get_tuple({"configurable": {"thread_id": "t2"}})
    assert len(state.checkpoint["channel_values"]["history"]) == 2


def test_unanswerable_question_abstains_with_nearest_sources(graph_env):
    svc, settings, _ = graph_env
    a = ask(svc, settings, "t3", "What is Acme Robotics' stock ticker symbol?")["answer"]
    assert a.abstained and a.text.startswith("I don't know")
    assert a.nearest_sources


def test_low_relevance_gate_skips_the_llm(graph_env):
    svc, settings, fake = graph_env
    strict = settings.model_copy(update={"abstain_dense_threshold": 0.99})
    fake.calls.clear()
    a = ask(svc, strict, "t4", "Quantum chromodynamics lattice gauge theory?")["answer"]
    assert a.abstained and a.abstain_reason == "low_relevance"
    assert "synthesis" not in fake.calls


def test_scope_with_no_documents(graph_env):
    svc, settings, _ = graph_env
    a = ask(svc, settings, "t5", "Anything?", scope_doc_ids=[])["answer"]
    assert a.abstained and a.abstain_reason == "no_documents"


def test_streaming_emits_status_and_tokens(graph_env):
    svc, settings, _ = graph_env
    types = [
        e["type"] for e in stream_turn(svc, settings, "t6", "When are production deploys frozen?")
    ]
    assert "status" in types and "token" in types and types[-1] == "done"


def test_router_picks_code_agent_for_symbols(graph_env):
    svc, settings, _ = graph_env
    ctx = AgentContext(svc, settings)
    docs = svc.catalog.ready_documents()
    from docchat.agents.retrieval import probe

    q = "Is there a bug in the pagination of list_items?"
    plan = plan_question(ctx, q, docs, probe(ctx, q, None))
    assert "code" in plan.agents and plan.symbols == ["list_items"]
    q2 = "When are production deploys frozen?"
    assert plan_question(ctx, q2, docs, probe(ctx, q2, None)).agents == ["retrieval"]


def test_mentioned_documents(graph_env):
    svc, _, _ = graph_env
    docs = svc.catalog.ready_documents()
    names = [
        d.file_name
        for d in mentioned_documents(
            "compare the engineering handbook and the meeting notes 2025-11-03", docs
        )
    ]
    assert "engineering_handbook.md" in names and "meeting_notes_2025-11-03.txt" in names


def test_mermaid_diagram_lists_agents():
    diagram = mermaid()
    for name in ("retrieve", "table_agent", "vision_agent", "code_agent", "verify"):
        assert name in diagram


def _hallucinating(tag, messages, system):
    from tests.fakes import default_responder

    if tag == "synthesis":
        return ("Production deploys are frozen from December 20 to January 5 [S1]. "
                "The freeze affects 97 engineering teams [S1].")  # fmt: skip
    if tag == "revise":
        return "Production deploys are frozen from December 20 to January 5 [S1]."
    return default_responder(tag, messages, system)


@pytest.mark.parametrize("mode", ["balanced", "deep"])
def test_unsupported_claim_is_removed_or_revised(graph_env, mode):
    svc, settings, fake = graph_env
    fake.responder = _hallucinating
    settings = settings.model_copy(update={"answer_mode": mode})
    done = ask(svc, settings, f"h-{mode}", "When are production deploys frozen?")
    a = done["answer"]
    assert "97" not in a.text and "December 20" in a.text
    verify_trace = next(t for t in done["trace"] if t["node"] == "verify")
    if mode == "deep":
        assert "revise" in fake.calls and verify_trace.get("revised")
        assert a.removed_sentences == []
    else:
        assert "revise" not in fake.calls
        assert any("97" in s for s in a.removed_sentences)
