"""Command line: ``uv run docchat [ui|ingest|ask|doctor|eval|prefetch|graph]``.

With no sub-command the Streamlit UI starts. Note: the embedded Qdrant store can only be opened
by one process, so stop the UI before using ``ingest`` / ``ask`` / ``eval`` (or run a Qdrant
server and set DOCCHAT_QDRANT_URL).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from uuid import uuid4

COMMANDS = {"ui", "ingest", "ask", "doctor", "eval", "prefetch", "graph"}


def _ui(extra: list[str]) -> int:
    from streamlit.web import cli as stcli

    app = Path(__file__).parent / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app), *extra]
    return stcli.main()


def _services():
    from docchat.config import get_settings
    from docchat.services import Services

    settings = get_settings()
    return Services(settings), settings


def _ingest(args) -> int:
    from docchat.ingest.detect import detect_format
    from docchat.ingest.pipeline import ingest_file

    services, settings = _services()
    paths: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        paths.extend(
            sorted(f for f in p.rglob("*") if f.is_file() and detect_format(f))
            if p.is_dir()
            else [p]
        )
    failed = 0
    try:
        for p in paths:
            r = ingest_file(
                services, settings, p, lambda m: print(f"  {m}", flush=True), args.force
            )
            mark = {"ready": "OK ", "skipped": "== ", "error": "ERR"}[r.status]
            print(f"{mark} {r.file_name}: {r.error or f'{r.n_chunks} chunks, {r.n_tables} tables'}"
                  + (f" ({r.seconds:.1f}s)" if r.seconds else ""))  # fmt: skip
            failed += r.status == "error"
    finally:
        services.close()
    return 1 if failed else 0


def _ask(args) -> int:
    from docchat.graph.run import stream_turn

    services, settings = _services()
    if args.mode:
        settings = settings.model_copy(update={"answer_mode": args.mode})
    thread = args.thread or uuid4().hex
    try:
        for q in args.question:
            final = None
            for event in stream_turn(services, settings, thread, q):
                if event["type"] == "status" and not args.json:
                    print(f"  · {event['text']}", file=sys.stderr, flush=True)
                elif event["type"] == "done":
                    final = event
            answer = final["answer"]
            if args.json:
                print(json.dumps({"question": q, "answer": answer.model_dump(), "plan": final["plan"],
                                  "llm_calls": final["llm_calls"], "seconds": final["seconds"]},
                                 indent=2, default=str))  # fmt: skip
                continue
            print(f"\nQ: {q}\n\n{answer.text}\n")
            for c in answer.citations:
                print(f"  [{c.n}] {c.file_name} {c.location}  ({c.agent})")
            if answer.confidence:
                print(f"\n  confidence: {answer.confidence.band} ({answer.confidence.score:.2f})"
                      f" · {final['seconds']:.1f}s · {len(final['llm_calls'])} LLM calls")  # fmt: skip
    finally:
        services.close()
    return 0


def _eval(args) -> int:
    from docchat.eval.run import run_eval

    services, settings = _services()
    try:
        report = run_eval(services, settings, Path(args.golden), k=args.k,
                          answers=not args.retrieval_only, judge_model=args.judge,
                          limit=args.limit, include_large=args.large)  # fmt: skip
    finally:
        services.close()
    print(json.dumps(report["summary"], indent=2))
    print(f"\nReport written to {report['path']}")
    return 0


def _graph(_args) -> int:
    from docchat.graph.run import mermaid

    print(mermaid())
    return 0


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] not in COMMANDS | {"-h", "--help"}:
        sys.exit(_ui(argv))  # `docchat` / `docchat --server.port 8600` -> UI
    parser = argparse.ArgumentParser(prog="docchat", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ui", help="start the Streamlit app (default)")
    p = sub.add_parser("ingest", help="index files or folders")
    p.add_argument("paths", nargs="+")
    p.add_argument("--force", action="store_true", help="re-index even if unchanged")
    p = sub.add_parser("ask", help="ask one or more questions (same thread = follow-ups)")
    p.add_argument("question", nargs="+")
    p.add_argument("--thread", help="conversation id (reuse for follow-ups across runs)")
    p.add_argument("--mode", choices=["fast", "balanced", "deep"])
    p.add_argument("--json", action="store_true")
    sub.add_parser("doctor", help="check Ollama, models, stores and optional backends")
    p = sub.add_parser("eval", help="evaluate on a golden Q&A set")
    p.add_argument("--golden", default="demo_corpus/golden.jsonl")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--retrieval-only", action="store_true", help="skip answer generation")
    p.add_argument("--judge", help="Ollama model used as LLM judge (e.g. gemma4:12b)")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--large", action="store_true", help="include questions on the large PDF")
    sub.add_parser("prefetch", help="download reranker / OCR models for offline use")
    sub.add_parser("graph", help="print the agent graph as Mermaid")

    if argv[0] == "ui":
        sys.exit(_ui(argv[1:]))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")  # fmt: skip
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    if args.cmd == "doctor":
        from docchat.doctor import run_doctor

        sys.exit(run_doctor())
    if args.cmd == "prefetch":
        from docchat.doctor import prefetch

        sys.exit(prefetch())
    handler = {"ingest": _ingest, "ask": _ask, "eval": _eval, "graph": _graph}[args.cmd]
    sys.exit(handler(args))
