"""Full-stack tests against a real Ollama (run on the demo machine):

    uv run pytest -m ollama

They ingest the whole demo corpus with the configured models (BGE-M3, qwen3.5 / gemma4,
reranker, OCR) and check real answers for each capability the brief asks for.
"""

from __future__ import annotations

import pandas as pd
import pytest

from docchat.agents.verification import numbers
from docchat.config import Settings
from docchat.graph.run import ask
from docchat.ingest.pipeline import ingest_file
from docchat.llm import check_ollama
from docchat.services import Services

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def stack(tmp_path_factory, corpus):
    settings = Settings(data_dir=tmp_path_factory.mktemp("stack"), caption_images=True)
    status = check_ollama(settings)
    if (
        not status.reachable
        or not status.has(settings.llm_model)
        or not status.has(settings.embed_model)
    ):
        pytest.skip(f"Ollama with {settings.llm_model} and {settings.embed_model} required")
    svc = Services(settings)
    for f in sorted(corpus.iterdir()):
        if f.suffix != ".jsonl" and f.name != "manifest.json":
            r = ingest_file(svc, settings, f)
            assert r.status == "ready", (f.name, r.error)
    yield svc, settings
    svc.close()


def _ask(stack, q, thread="t"):
    svc, settings = stack
    return ask(svc, settings, thread, q)["answer"]


def test_json_probe(stack):
    svc, _ = stack
    assert svc.llm.probe_json()


def test_pdf_lookup_cites_page(stack):
    a = _ask(stack, "What was Acme Robotics' total revenue in FY2025?")
    assert "48.6" in numbers(a.text)
    assert any(c.file_name == "acme_annual_report_2025.pdf" and c.page == 1 for c in a.citations)


def test_spreadsheet_sql(stack, corpus):
    orders = pd.read_excel(corpus / "sales_2025.xlsx", sheet_name="Orders")
    expected = str(int(orders.loc[orders.region == "Europe", "revenue"].sum()))
    a = _ask(stack, "What is the total revenue for Europe in the Orders sheet?")
    assert expected in numbers(a.text)
    assert any(c.agent == "table" for c in a.citations)


def test_scanned_pdf_ocr(stack):
    a = _ask(stack, "What is the total contract value of the Kestrel supply agreement?")
    assert "1250000" in numbers(a.text)


def test_chart_vision(stack):
    a = _ask(stack, "According to Figure 1 in the annual report, what was Q3 revenue?")
    assert "12.4" in numbers(a.text)


def test_code_bug(stack):
    a = _ask(stack, "Is there a bug in the pagination of inventory_service.py?")
    assert "list_items" in a.text and any(
        c.file_name == "inventory_service.py" for c in a.citations
    )


def test_cross_document_conflict(stack):
    a = _ask(stack, "How many employees does Acme Robotics have?")
    assert {"412", "398"} <= numbers(a.text)


def test_followup_memory(stack):
    _ask(stack, "What was revenue by region in FY2025?", thread="fu")
    a = _ask(stack, "And which of those regions grew the slowest?", thread="fu")
    assert "LATAM" in a.text


def test_graceful_i_dont_know(stack):
    a = _ask(stack, "What is Acme Robotics' stock ticker symbol?")
    assert a.abstained
