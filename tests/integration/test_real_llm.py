"""Tests against a REAL local LLM (Ollama). Not run by default (`-m ollama` is excluded).

    uv run pytest -m ollama tests/integration/test_real_llm.py -v -s

What is covered, in three layers:

1. **LLM contract** (runs once per model): capabilities, clean replies, streaming, thinking mode,
   schema-constrained JSON (repeated), vision on an image with known content, chart reading,
   long prompts are not truncated (num_ctx is really applied), concurrent JSON calls.
2. **Embeddings**: dimension, finite vectors on hostile inputs, semantic ordering.
3. **Agents with the real model** (isolated: the hashing embedder is used so only the LLM is
   under test): text-to-SQL exact answers on the golden spreadsheet questions, SQL follow-up,
   vision agent on a chart, image captioning, follow-up rewriting, planner decomposition,
   synthesis citations + NOT_FOUND, LLM claim verification, deep-mode revision.
4. **Optional golden-set run** of the whole system (real embeddings + reranker + OCR) with
   accuracy thresholds: set ``DOCCHAT_TEST_GOLDEN=1``.

Environment variables:
    DOCCHAT_TEST_MODELS        comma-separated chat models for layer 1, e.g. "qwen3.5:4b,gemma4:e4b"
                               (default: DOCCHAT_LLM_MODEL). Models that are not pulled are skipped.
    DOCCHAT_TEST_GOLDEN=1      also run the full golden set (slow; needs the native libraries)
    DOCCHAT_TEST_MIN_ACCURACY  answer-accuracy threshold for the golden run (default 0.6)
"""

from __future__ import annotations

import math
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from docchat.agents import memory, synthesis, verification
from docchat.agents import table as table_agent
from docchat.agents import vision as vision_agent
from docchat.agents.router import _llm_decompose
from docchat.agents.types import AgentContext, Evidence, Plan, SubQuery, Verdict
from docchat.config import Settings, get_settings
from docchat.eval.golden import load_golden
from docchat.index.embeddings import OllamaEmbedder
from docchat.ingest.context import ParseContext
from docchat.ingest.pipeline import caption_figures, ingest_file
from docchat.llm import LLM, check_ollama, get_client
from docchat.services import Services

pytestmark = [pytest.mark.ollama, pytest.mark.timeout(1800)]

MODELS = [m.strip() for m in os.environ.get("DOCCHAT_TEST_MODELS", "").split(",") if m.strip()] or [
    get_settings().llm_model
]
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


# ------------------------------------------------------------------------------------------------
# helpers & fixtures
# ------------------------------------------------------------------------------------------------
def _floats(text: str) -> list[float]:
    out = []
    for raw in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def result_part(evidence_content: str) -> str:
    """Only the computed result of a table-agent evidence (not the file name or the SQL text)."""
    m = re.search(r"(RESULT:|Result \()", evidence_content)
    return evidence_content[m.start() :] if m else evidence_content


def has_number(text: str, expected: str) -> bool:
    """True if ``text`` contains ``expected`` up to the precision it is written with (4.9 ~ 4.8923)."""
    target = float(expected.replace(",", ""))
    decimals = len(expected.split(".")[1]) if "." in expected else 0
    tol = 0.5 * 10**-decimals + 1e-9
    return any(abs(x - target) <= tol for x in _floats(text))


@pytest.fixture(scope="module")
def status():
    st = check_ollama(get_settings())
    if not st.reachable:
        pytest.skip(f"Ollama is not reachable at {get_settings().ollama_host}")
    return st


@pytest.fixture(scope="module", params=MODELS)
def llm(request, status) -> LLM:
    model = request.param
    if not status.has(model):
        pytest.skip(f"{model} is not pulled (ollama pull {model})")
    settings = get_settings().model_copy(update={"llm_model": model})
    return LLM(get_client(settings), settings, model)


@pytest.fixture(scope="module")
def agent_env(tmp_path_factory, corpus, status):
    """Real LLM, hashing embedder (so failures point at the LLM), a few ingested files."""
    model = MODELS[0]
    if not status.has(model):
        pytest.skip(f"{model} is not pulled")
    settings = Settings(
        data_dir=tmp_path_factory.mktemp("real_llm"),
        llm_model=model,
        embed_backend="hashing",
        reranker_backend="none",
        caption_images=False,
    )
    services = Services(settings)
    docs = {}
    for name in ("sales_2025.xlsx", "support_tickets.csv", "engineering_handbook.md",
                 "monthly_active_robots.png"):  # fmt: skip
        result = ingest_file(services, settings, corpus / name)
        assert result.status == "ready", (name, result.error)
        docs[name] = result.doc_id
    yield AgentContext(services, settings), docs
    services.close()


def _tables(ctx: AgentContext, doc_id: str) -> list[str]:
    return [t.name for t in ctx.services.catalog.tables_for(doc_id)]


HANDBOOK_EVIDENCE = Evidence(
    id="hb", agent="retrieval", kind="text", score=0.9, doc_id="hb", file_name="engineering_handbook.md",
    location="L7-15",
    content=("Primary on-call rotates weekly. Handover happens on Mondays at 10:00 CET. "
             "Pages must be acknowledged within 15 minutes. "
             "Production deploys are frozen from December 20 to January 5."),
)  # fmt: skip


# ------------------------------------------------------------------------------------------------
# 1. LLM contract (per model)
# ------------------------------------------------------------------------------------------------
def test_capabilities_and_context(llm: LLM):
    assert "completion" in llm.capabilities
    assert 2048 <= llm.num_ctx <= llm.settings.num_ctx
    print(f"\n{llm.model}: capabilities={sorted(llm.capabilities)} num_ctx={llm.num_ctx}")


def test_plain_reply_is_clean(llm: LLM):
    res = llm.chat([{"role": "user", "content": "Reply with exactly one word: pong"}], think=False,
                   temperature=0.0, num_predict=20)  # fmt: skip
    assert "pong" in res.content.lower()
    assert "<think>" not in res.content
    assert res.prompt_tokens > 0 and res.output_tokens > 0
    assert res.done_reason == "stop"


def test_streaming_matches_final_text(llm: LLM):
    tokens: list[str] = []
    res = llm.chat([{"role": "user", "content": "Count from 1 to 10, separated by spaces."}],
                   think=False, temperature=0.0, num_predict=60, on_token=tokens.append)  # fmt: skip
    assert len(tokens) > 1
    assert "".join(tokens).strip() == res.content.strip()
    assert "10" in res.content


def test_thinking_mode(llm: LLM):
    if not llm.supports_thinking:
        pytest.skip(f"{llm.model} has no thinking capability")
    res = llm.chat([{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
                   think=True, temperature=0.0, num_predict=2048)  # fmt: skip
    assert "391" in res.content
    assert "<think>" not in res.content  # reasoning is separated from the answer


class Extraction(BaseModel):
    company: str
    year: int
    revenue_musd: float
    sentiment: Literal["positive", "negative", "neutral"]


def test_structured_json_is_reliable(llm: LLM):
    text = ("In FY2025 Acme Robotics had a record year: revenue grew to USD 48.6 million "
            "and margins improved.")  # fmt: skip
    for _ in range(5):  # small models are tested for consistency, not a single lucky answer
        out = llm.json([{"role": "user", "content": f"Extract the fields from: {text}"}], Extraction,
                       tag="test-json", num_predict=200)  # fmt: skip
        assert "acme" in out.company.lower()
        assert out.year == 2025
        assert math.isclose(out.revenue_musd, 48.6, abs_tol=0.01)
        assert out.sentiment == "positive"


def test_probe_json_uses_schema_mode(llm: LLM):
    assert llm.probe_json(), "schema-constrained output failed; update Ollama to >= 0.34.4"
    assert llm.json_mode == "schema"


def _font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def test_vision_reads_text_in_an_image(llm: LLM, tmp_path: Path):
    if not llm.supports_vision:
        pytest.skip(f"{llm.model} has no vision capability")
    img = Image.new("RGB", (900, 300), "white")
    ImageDraw.Draw(img).text((40, 100), "INVOICE TOTAL: 7391", fill="black", font=_font(64))
    path = tmp_path / "digits.png"
    img.save(path)
    res = llm.chat([{"role": "user", "content": "What number is written in this image? Reply with the number only."}],
                   images=[path], think=False, temperature=0.0, num_predict=30)  # fmt: skip
    assert "7391" in res.content.replace(",", "")


def test_vision_reads_a_chart(llm: LLM, corpus: Path):
    if not llm.supports_vision:
        pytest.skip(f"{llm.model} has no vision capability")
    res = llm.chat(
        [{"role": "user", "content": "This line chart shows monthly values. In which month did the "
                                     "value go DOWN compared with the previous month? Answer with the month."}],
        images=[corpus / "monthly_active_robots.png"], think=False, temperature=0.0, num_predict=60,
    )  # fmt: skip
    assert "sep" in res.content.lower()


def test_long_prompt_is_not_truncated(llm: LLM):
    """Ollama's default window is 4K and it silently drops the start of longer prompts.
    The needle is at the very beginning, so this passes only if num_ctx is really applied."""
    if llm.num_ctx < 8192:
        pytest.skip("num_ctx < 8192")
    filler = (
        "The quarterly review covered routine operational topics and no decisions were made. " * 450
    )
    prompt = f"The secret code is 4417.\n\n{filler}\n\nWhat is the secret code mentioned at the very beginning? Reply with the code only."
    res = llm.chat(
        [{"role": "user", "content": prompt}], think=False, temperature=0.0, num_predict=20
    )
    assert res.prompt_tokens > 4096, res.prompt_tokens
    assert "4417" in res.content


def test_concurrent_json_calls(llm: LLM):
    class Answer(BaseModel):
        value: int

    questions = [
        f"What is {a} + {b}? Reply as JSON." for a, b in [(2, 3), (10, 7), (40, 2), (9, 9)]
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda q: llm.json([{"role": "user", "content": q}], Answer, num_predict=60),
                questions,
            )
        )
    assert [r.value for r in results] == [5, 17, 42, 18]


# ------------------------------------------------------------------------------------------------
# 2. Embeddings
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def embedder(status):
    s = get_settings()
    if not status.has(s.embed_model):
        pytest.skip(f"{s.embed_model} is not pulled")
    return OllamaEmbedder(get_client(s), s.embed_model)


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_embedding_dimension_and_semantics(embedder):
    if embedder.name.startswith("bge-m3"):
        assert embedder.dim == 1024
    q = embedder.embed_query("How many people work at the company?")
    pos, neg = embedder.embed_documents(["Acme employs 412 staff across three offices.",
                                         "Production deploys are frozen in late December."])  # fmt: skip
    assert _cos(q, pos) > _cos(q, neg)


def test_embeddings_survive_hostile_inputs(embedder):
    texts = [
        "x" * 30000,
        " ",
        "émojis 🚀📄 ünïcödé",
        "def f(x):\n    return x ** 2\n" * 200,
        "| a | b |\n|---|---|\n" * 300,
    ]
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == len(texts)
    for v in vectors:
        assert len(v) == embedder.dim and all(math.isfinite(x) for x in v)


# ------------------------------------------------------------------------------------------------
# 3. Agents with the real model
# ------------------------------------------------------------------------------------------------
GOLDEN_TABLE_IDS = ["xl-01", "xl-02", "xl-03", "xl-04", "xl-05", "csv-01", "csv-02"]


@pytest.mark.parametrize("item_id", GOLDEN_TABLE_IDS)
def test_text_to_sql_matches_golden(agent_env, corpus, item_id):
    ctx, docs = agent_env
    item = next(i for i in load_golden(corpus / "golden.jsonl") if i.id == item_id)
    doc = docs["sales_2025.xlsx"] if item_id.startswith("xl") else docs["support_tickets.csv"]
    evidence, trace = table_agent.run(ctx, item.question, _tables(ctx, doc))
    assert evidence, f"no result; attempts: {trace.get('attempts')}"
    content = evidence[0].content
    result = result_part(content)
    for n in item.expected_numbers:
        assert has_number(result, n), (
            f"{item.question}: expected {n}\n{content}\n{trace.get('attempts')}"
        )
    for kw in item.expected_keywords:
        assert kw.lower() in result.lower(), f"{item.question}: expected {kw!r}\n{content}"
    print(
        f"\n{item_id}: {trace.get('method')} retries={trace.get('retries')} sql={trace.get('sql')}"
    )


def test_sql_followup_modifies_previous_query(agent_env, corpus):
    ctx, docs = agent_env
    tables = _tables(ctx, docs["sales_2025.xlsx"])
    first, _ = table_agent.run(
        ctx, "How many orders were placed in North America in Q4 2025?", tables
    )
    assert first
    last_sql = {"sql": first[0].sql, "tables": [first[0].table_name]}
    follow, trace = table_agent.run(ctx, "Same question, but for Europe.", tables, last_sql)
    orders = pd.read_excel(corpus / "sales_2025.xlsx", sheet_name="Orders")
    expected = int(
        ((orders.region == "Europe") & (pd.to_datetime(orders.order_date).dt.quarter == 4)).sum()
    )
    assert follow and has_number(result_part(follow[0].content), str(expected)), (
        expected,
        trace.get("attempts"),
    )


def test_vision_agent_reads_chart(agent_env):
    ctx, docs = agent_env
    figures = ctx.services.store.document_chunks(
        docs["monthly_active_robots.png"], kinds=["figure"]
    )
    assert figures
    evidence, trace = vision_agent.run(ctx, "In which month did the number of monthly active robots decrease?",
                                       [figures[0].chunk_id])  # fmt: skip
    assert evidence, trace
    assert "sep" in evidence[0].content.lower(), evidence[0].content


def test_image_captioning_extracts_chart_content(agent_env, corpus, tmp_path):
    ctx, _ = agent_env
    from docchat.ingest.parsers.image import parse_image

    settings = ctx.settings.model_copy(update={"caption_images": True})
    parsed = parse_image(
        corpus / "monthly_active_robots.png", ParseContext(settings, "cap", tmp_path)
    )
    caption_figures(ctx.services, settings, parsed, lambda m: None)
    text = parsed.elements[0].text.lower()
    assert "description (" in text, parsed.warnings
    assert any(m in text for m in MONTHS) or "1184" in text.replace(",", "")


def test_followup_rewrite(agent_env):
    ctx, _ = agent_env
    history = [
        {"q": "How many support tickets are in the log?", "a": "There are 250 support tickets [1]."}
    ]
    standalone = memory.rewrite(ctx, "And how many of those are critical?", history, "").lower()
    assert "ticket" in standalone and "critical" in standalone, standalone


def test_planner_decomposes_multi_part_question(agent_env):
    ctx, _ = agent_env
    q = ("What was the total revenue for Europe in the Orders sheet, and which product sold the most "
         "units in APAC, and when are production deploys frozen?")  # fmt: skip
    plan = Plan(agents=["retrieval"], sub_queries=[SubQuery(query=q)])
    _llm_decompose(ctx, q, plan)
    assert plan.used_llm
    assert len(plan.sub_queries) >= 2, plan.sub_queries


def test_synthesis_cites_sources(agent_env):
    ctx, _ = agent_env
    messages = synthesis.build_messages(
        "When are production deploys frozen?", [HANDBOOK_EVIDENCE], "lookup", ""
    )
    res = synthesis.write_answer(ctx, messages, think=False)
    assert "[S1]" in res.content, res.content
    assert "december 20" in res.content.lower()


def test_synthesis_says_not_found(agent_env):
    ctx, _ = agent_env
    messages = synthesis.build_messages(
        "What is Acme's stock ticker symbol?", [HANDBOOK_EVIDENCE], "lookup", ""
    )
    res = synthesis.write_answer(ctx, messages, think=False)
    assert synthesis.is_not_found(res.content), res.content


def test_llm_verifier_catches_contradiction(agent_env):
    ctx, _ = agent_env
    draft = ("Pages must be acknowledged within one hour [S1]. "
             "Engineers hand over the on-call duty every Monday [S1].")  # fmt: skip
    _, verdicts, stats = verification.verify(ctx, draft, [HANDBOOK_EVIDENCE], use_llm=True)
    assert stats["llm_checked"] >= 1
    assert verdicts[0].label != "supported", verdicts[0]
    assert verdicts[1].label != "unsupported", verdicts[1]


def test_revision_removes_flagged_claim(agent_env):
    ctx, _ = agent_env
    draft = ("Production deploys are frozen from December 20 to January 5 [S1]. "
             "The freeze affects 97 engineering teams [S1].")  # fmt: skip
    flagged = [Verdict(sentence="The freeze affects 97 engineering teams.", label="unsupported",
                       reason="number(s) 97 not found in the cited source")]  # fmt: skip
    revised = verification.revise(
        ctx, "When are production deploys frozen?", draft, flagged, [HANDBOOK_EVIDENCE]
    )
    assert revised and "97" not in revised and "december 20" in revised.lower(), revised


# ------------------------------------------------------------------------------------------------
# 4. Optional: the whole system on the golden set
# ------------------------------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("DOCCHAT_TEST_GOLDEN") != "1", reason="set DOCCHAT_TEST_GOLDEN=1"
)
def test_golden_set_end_to_end(tmp_path_factory, corpus, status):
    from docchat.eval.run import run_eval

    settings = Settings(data_dir=tmp_path_factory.mktemp("golden"))
    for model in (settings.llm_model, settings.embed_model):
        if not status.has(model):
            pytest.skip(f"{model} is not pulled")
    services = Services(settings)
    try:
        report = run_eval(services, settings, corpus / "golden.jsonl", k=5)
    finally:
        services.close()
    keep = Path("data") / "eval"
    keep.mkdir(parents=True, exist_ok=True)
    shutil.copy(report["path"], keep)  # visible in the app under "How it works"
    summary = report["summary"]
    print("\n", summary)
    min_acc = float(os.environ.get("DOCCHAT_TEST_MIN_ACCURACY", "0.6"))
    assert summary["answer_accuracy"] >= min_acc, summary
    assert summary["abstain_recall"] >= 0.66, summary
    assert summary.get("citation_precision") is None or summary["citation_precision"] >= 0.6, (
        summary
    )
