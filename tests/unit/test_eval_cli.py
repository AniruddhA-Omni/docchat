import json
import sys

from docchat.eval.golden import load_golden
from docchat.eval.run import _ece, run_eval
from docchat.services import Services
from tests.fakes import FakeLLM


def test_golden_set_is_valid(corpus):
    items = load_golden(corpus / "golden.jsonl")
    assert len(items) >= 30
    assert {
        "lookup",
        "table",
        "ocr",
        "vision",
        "code",
        "cross_doc",
        "followup",
        "unanswerable",
    } <= {i.type for i in items}
    files = {p.name for p in corpus.iterdir()}
    for item in items:
        for src in item.must_cite:
            assert src.file in files, (item.id, src.file)


def test_eval_harness_end_to_end(settings, corpus, monkeypatch):
    # Only formats that parse without native libraries on every machine.
    keep = {"engineering_handbook.md", "support_tickets.csv", "meeting_notes_2025-11-03.txt"}
    items = [i for i in load_golden(corpus / "golden.jsonl")
             if i.type == "unanswerable" or (i.must_cite and {s.file for s in i.must_cite} <= keep)]  # fmt: skip
    golden = corpus / "golden_subset.jsonl"
    golden.write_text("\n".join(i.model_dump_json() for i in items), encoding="utf-8")
    import docchat.eval.run as run_mod

    monkeypatch.setattr(run_mod, "ensure_corpus", lambda svc, st, d, large: [
        run_mod.ingest_file(svc, st, d / f) for f in sorted(keep)])  # fmt: skip
    settings = settings.model_copy(update={"abstain_dense_threshold": 0.1})
    svc = Services(settings, llm=FakeLLM())
    try:
        report = run_eval(svc, settings, golden, k=5)
    finally:
        svc.close()
    s = report["summary"]
    assert s["retrieval_hit@5"] > 0.5
    assert s["abstain_recall"] == 1.0
    assert {r["mode"] for r in report["ablation"]} == {"dense", "bm25", "hybrid"}
    from pathlib import Path

    saved = json.loads(Path(report["path"]).read_text(encoding="utf-8"))
    assert saved["summary"] == s


def test_ece():
    assert _ece([(0.9, True), (0.9, True), (0.1, False)]) < 0.15
    assert _ece([(0.9, False), (0.9, False)]) > 0.8


def test_cli_graph_command(capsys, monkeypatch):
    from docchat import cli

    monkeypatch.setattr(sys, "argv", ["docchat", "graph"])
    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 0
    assert "table_agent" in capsys.readouterr().out
