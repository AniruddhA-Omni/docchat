"""Evaluate DocChat on a golden Q&A set.

Retrieval (every question with an expected source): hit@k and MRR for dense-only, BM25-only,
hybrid (RRF) and hybrid + rerank. Answers (full graph): correctness (expected numbers and
keywords present), citation precision/recall at file and page level, table/SQL exact match,
abstention precision/recall on unanswerable questions, confidence calibration (ECE) and latency.
Optionally an LLM judge (a different local model) grades answers against the reference.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from docchat.agents import retrieval
from docchat.agents.types import AgentContext, AnswerPayload
from docchat.agents.verification import numbers
from docchat.config import Settings
from docchat.eval.golden import GoldenItem, load_golden
from docchat.graph.run import ask
from docchat.ingest.detect import detect_format
from docchat.ingest.pipeline import ingest_file
from docchat.schemas import Chunk
from docchat.services import Services

MODES = [("dense", "dense", False), ("bm25", "sparse", False), ("hybrid", "hybrid", False),
         ("hybrid+rerank", "hybrid", True)]  # fmt: skip


class Judgement(BaseModel):
    verdict: str  # correct | partial | incorrect
    reason: str = ""


JUDGE_SYSTEM = (
    "You grade an assistant's answer against a reference answer. Reply as JSON "
    '{"verdict": "correct" | "partial" | "incorrect", "reason": "..."}. '
    "'correct' = same facts/numbers as the reference (wording may differ); 'partial' = some "
    "correct facts but something important missing or wrong; 'incorrect' otherwise. For questions "
    "whose reference says the information is not in the documents, a refusal is correct."
)


def _matches(chunk_file: str, page_start: int | None, page_end: int | None, exp_file: str,
             exp_page: int | None, approx: bool = False) -> bool:  # fmt: skip
    if chunk_file != exp_file:
        return False
    if exp_page is None or page_start is None:
        return True
    slack = 1 if approx else 0
    return page_start - slack <= exp_page <= (page_end or page_start) + slack


def _chunk_hit(c: Chunk, item: GoldenItem) -> bool:
    return any(_matches(c.file_name, c.page_start, c.page_end, e.file, e.page, c.page_approx)
               for e in item.must_cite)  # fmt: skip


def ensure_corpus(
    services: Services, settings: Settings, corpus_dir: Path, include_large: bool
) -> None:
    for f in sorted(corpus_dir.iterdir()):
        if not f.is_file() or detect_format(f) is None or f.name in ("manifest.json",):
            continue
        if f.name == "acme_policies_large.pdf" and not include_large:
            continue
        r = ingest_file(services, settings, f, lambda m: None)
        if r.status == "error":
            print(f"  ! {f.name}: {r.error}")


def eval_retrieval(ctx: AgentContext, items: list[GoldenItem], k: int) -> list[dict]:
    rows = []
    for label, mode, rerank in MODES:
        if rerank and ctx.services.reranker is None:
            continue
        hits, rr, n = 0, 0.0, 0
        for item in items:
            if not item.answerable or not item.must_cite or item.history:
                continue
            n += 1
            found = retrieval.search(ctx, item.question, None, top_n=k, mode=mode, rerank=rerank)
            ranks = [i for i, h in enumerate(found, 1) if _chunk_hit(h.chunk, item)]
            hits += bool(ranks)
            rr += 1 / ranks[0] if ranks else 0.0
        if n:
            rows.append(
                {"mode": label, f"hit@{k}": round(hits / n, 3), "mrr": round(rr / n, 3), "n": n}
            )
    return rows


def _answer_metrics(item: GoldenItem, a: AnswerPayload) -> dict:
    text_nums = numbers(a.text)
    nums_ok = all(n.replace(",", "") in text_nums for n in item.expected_numbers)
    kw_ok = all(kw.lower() in a.text.lower() for kw in item.expected_keywords)
    cited_files = {c.file_name for c in a.citations}
    expected_files = {e.file for e in item.must_cite}
    page_hits = [e for e in item.must_cite if e.page and any(
        _matches(c.file_name, c.page, c.page, e.file, e.page, c.page_approx) for c in a.citations)]  # fmt: skip
    pages_expected = [e for e in item.must_cite if e.page]
    return {
        "correct": (
            a.abstained if not item.answerable else (not a.abstained and nums_ok and kw_ok)
        ),
        "numbers_ok": nums_ok,
        "keywords_ok": kw_ok,
        "cite_precision": (len(cited_files & expected_files) / len(cited_files))
        if cited_files and expected_files
        else None,
        "cite_recall": (len(cited_files & expected_files) / len(expected_files))
        if expected_files
        else None,
        "page_recall": (len(page_hits) / len(pages_expected)) if pages_expected else None,
    }


def _judge(
    services: Services, settings: Settings, model: str, item: GoldenItem, a: AnswerPayload
) -> str:
    llm = services.llm.with_model(model)
    prompt = (
        f"Question: {item.question}\nReference answer: {item.reference}\nAssistant answer: {a.text}"
    )
    try:
        return llm.json([{"role": "user", "content": prompt}], Judgement, system=JUDGE_SYSTEM,
                        tag="judge", num_predict=200).verdict.lower()  # fmt: skip
    except Exception:
        return "error"


def _ece(pairs: list[tuple[float, bool]], bins: int = 5) -> float | None:
    if not pairs:
        return None
    total, err = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        group = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if group:
            conf = sum(c for c, _ in group) / len(group)
            acc = sum(ok for _, ok in group) / len(group)
            err += len(group) / total * abs(conf - acc)
    return round(err, 3)


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def run_eval(services: Services, settings: Settings, golden_path: Path, k: int = 5,
             answers: bool = True, judge_model: str | None = None, limit: int | None = None,
             include_large: bool = False) -> dict:  # fmt: skip
    items = [i for i in load_golden(golden_path) if include_large or not i.requires_large][:limit]
    print(f"Evaluating {len(items)} questions from {golden_path}")
    ensure_corpus(services, settings, golden_path.parent, include_large)
    ctx = AgentContext(services, settings)
    ablation = eval_retrieval(ctx, items, k)
    for row in ablation:
        print(f"  retrieval {row['mode']:<14} hit@{k}={row[f'hit@{k}']:.2f}  mrr={row['mrr']:.2f}")

    results = []
    if answers:
        for n, item in enumerate(items, 1):
            thread = f"eval-{uuid4().hex[:8]}"
            for prior in item.history:
                ask(services, settings, thread, prior)
            start = time.perf_counter()
            done = ask(services, settings, thread, item.question)
            seconds = time.perf_counter() - start
            a: AnswerPayload = done["answer"]
            row = {"id": item.id, "type": item.type, "question": item.question, **_answer_metrics(item, a),
                   "abstained": a.abstained, "confidence": a.confidence.score if a.confidence else None,
                   "seconds": round(seconds, 2), "llm_calls": len(done["llm_calls"]),
                   "agents": ",".join(done["plan"].get("agents", [])), "answer": a.text[:300]}  # fmt: skip
            if judge_model:
                row["judge"] = _judge(services, settings, judge_model, item, a)
            results.append(row)
            print(
                f"  [{n}/{len(items)}] {'✔' if row['correct'] else '✘'} {item.id} ({seconds:.1f}s)"
            )

    summary: dict = {}
    best = next((r for r in reversed(ablation)), None)
    if best:
        summary[f"retrieval_hit@{k}"] = best[f"hit@{k}"]
        summary["retrieval_mrr"] = best["mrr"]
    if results:
        answerable = [r for r in results if r["type"] != "unanswerable"]
        unanswerable = [r for r in results if r["type"] == "unanswerable"]
        tables = [r for r in results if r["type"] == "table"]
        abstained = [r for r in results if r["abstained"]]
        summary.update({
            "answer_accuracy": _mean(r["correct"] for r in answerable),
            "table_exact_match": _mean(r["numbers_ok"] for r in tables),
            "citation_precision": _mean(r["cite_precision"] for r in answerable),
            "citation_recall": _mean(r["cite_recall"] for r in answerable),
            "page_recall": _mean(r["page_recall"] for r in answerable),
            "abstain_recall": _mean(r["abstained"] for r in unanswerable),
            "abstain_precision": (round(sum(r["type"] == "unanswerable" for r in abstained) / len(abstained), 3)
                                  if abstained else None),
            "ece": _ece([(r["confidence"], r["correct"]) for r in answerable if r["confidence"] is not None]),
            "latency_p50_s": round(statistics.median(r["seconds"] for r in results), 2),
            "latency_p95_s": round(sorted(r["seconds"] for r in results)[int(0.95 * (len(results) - 1))], 2),
        })  # fmt: skip
        if judge_model:
            summary["judge_correct"] = _mean(r.get("judge") == "correct" for r in results)
        by_type: dict[str, list] = {}
        for r in results:
            by_type.setdefault(r["type"], []).append(r["correct"])
        summary["accuracy_by_type"] = {
            t: round(sum(v) / len(v), 2) for t, v in sorted(by_type.items())
        }

    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report = {"model": settings.llm_model, "embed_model": settings.embed_model,
              "reranker": services.reranker.name if services.reranker else None,
              "mode": settings.answer_mode, "k": k, "summary": summary, "ablation": ablation,
              "items": results, "path": str(path)}  # fmt: skip
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
