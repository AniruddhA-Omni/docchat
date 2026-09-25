"""``docchat doctor`` (environment checks with fixes) and ``docchat prefetch`` (offline models)."""

from __future__ import annotations

import importlib
import shutil
import subprocess
from dataclasses import dataclass

import httpx

from docchat.config import apply_runtime_env, get_settings
from docchat.extras import OCR_BACKENDS, PDF_PARSERS, backend_available, install_hint

MIN_OLLAMA = (0, 34, 4)  # structured outputs + think=false fixed for qwen3.5 / gemma4


@dataclass
class Check:
    name: str
    ok: bool | None  # None = informational / optional
    detail: str
    fix: str = ""


def _native(module: str, attr: str | None = None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
        if attr:
            getattr(mod, attr)
        return True, getattr(mod, "__version__", "") or "ok"
    except (ImportError, OSError) as exc:
        return False, str(exc).split("\n")[0][:120]


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("-")[0].split(".")[:3] if x.isdigit())


def checks() -> list[Check]:
    s = get_settings()
    apply_runtime_env(s)
    out: list[Check] = []
    msvc_fix = "Install the Microsoft Visual C++ Redistributable: winget install Microsoft.VCRedist.2015+.x64"

    # --- Ollama ---------------------------------------------------------------------------------
    try:
        version = httpx.get(f"{s.ollama_host}/api/version", timeout=5).json().get("version", "?")
        ok = _version_tuple(version) >= MIN_OLLAMA
        out.append(Check("Ollama server", ok, f"{s.ollama_host} v{version}",
                         "" if ok else "Update Ollama (>= 0.34.4) for reliable JSON with thinking models"))  # fmt: skip
        reachable = True
    except Exception as exc:
        out.append(Check("Ollama server", False, f"not reachable at {s.ollama_host} ({exc.__class__.__name__})",
                         "Start the Ollama app or run `ollama serve`"))  # fmt: skip
        reachable = False

    if reachable:
        from docchat.llm import LLM, check_ollama, get_client

        status = check_ollama(s)
        for model in dict.fromkeys([s.llm_model, s.effective_vision_model, s.embed_model]):
            out.append(Check(f"model {model}", status.has(model), "pulled" if status.has(model) else "missing",
                             f"ollama pull {model}"))  # fmt: skip
        if status.has(s.llm_model):
            llm = LLM(get_client(s), s)
            caps = ", ".join(sorted(llm.capabilities))
            out.append(Check("chat model capabilities", "vision" in llm.capabilities, caps,
                             "Pick a vision-capable model (qwen3.5 / gemma4) or set DOCCHAT_VISION_MODEL"))  # fmt: skip
            out.append(Check("num_ctx", None, f"{llm.num_ctx} tokens (model max respected)"))
            ok = llm.probe_json()
            out.append(Check("structured JSON output", ok, "schema-constrained" if ok else "fallback to format=json",
                             "" if ok else "Update Ollama; the app still works with the fallback"))  # fmt: skip
        if status.has(s.embed_model):
            from docchat.index.embeddings import OllamaEmbedder

            try:
                emb = OllamaEmbedder(get_client(s), s.embed_model)
                out.append(Check("embeddings", True, f"{s.embed_model}: {emb.dim} dimensions"))
            except Exception as exc:
                out.append(Check("embeddings", False, str(exc)[:120]))
        try:
            loaded = get_client(s).ps().models
            gpu = [m.model for m in loaded if (m.size_vram or 0) > 0]
            out.append(Check("Ollama GPU offload", None, ", ".join(gpu) + " on GPU" if gpu
                             else "no model currently loaded on GPU (CPU inference)"))  # fmt: skip
        except Exception:
            pass

    # --- storage --------------------------------------------------------------------------------
    if s.qdrant_url:
        try:
            httpx.get(f"{s.qdrant_url}/readyz", timeout=5).raise_for_status()
            out.append(Check("Qdrant", True, f"server at {s.qdrant_url}"))
        except Exception as exc:
            out.append(Check("Qdrant", False, f"{s.qdrant_url}: {exc.__class__.__name__}",
                             "docker compose up -d qdrant"))  # fmt: skip
    else:
        out.append(Check("Qdrant", None, f"embedded at {s.qdrant_path} (single process)"))

    # --- native libraries -----------------------------------------------------------------------
    ok, detail = _native("pymupdf")
    out.append(Check("PyMuPDF (PDF parsing, previews)", ok, detail, "" if ok else msvc_fix))
    ok, detail = _native("onnxruntime")
    out.append(Check("ONNX Runtime (OCR, reranker)", ok, detail, "" if ok else msvc_fix))
    ok, detail = _native("rapidocr", "RapidOCR")
    out.append(Check("RapidOCR", ok, detail if not ok else "ok", "" if ok else msvc_fix))
    from docchat.index.rerank import load_reranker

    rr = load_reranker(s)
    out.append(Check("reranker", rr is not None if s.reranker_backend != "none" else None,
                     rr.name if rr else f"{s.reranker_backend} unavailable",
                     "" if rr else "uv run docchat prefetch (needs internet once)"))  # fmt: skip

    # --- optional backends ----------------------------------------------------------------------
    for name in PDF_PARSERS:
        out.append(Check(f"PDF parser: {name}", None if name != s.pdf_parser else backend_available("pdf", name),
                         "installed" if backend_available("pdf", name) else "not installed",
                         "" if backend_available("pdf", name) else install_hint("pdf", name)))  # fmt: skip
    for name in OCR_BACKENDS:
        out.append(Check(f"OCR: {name}", None if name != s.ocr_backend else backend_available("ocr", name),
                         "installed" if backend_available("ocr", name) else "not installed",
                         "" if backend_available("ocr", name) else install_hint("ocr", name)))  # fmt: skip
    try:
        import torch

        cuda = torch.cuda.is_available()
        out.append(
            Check(
                "PyTorch",
                None,
                f"{torch.__version__}, CUDA {'available: ' + torch.cuda.get_device_name(0) if cuda else 'not available'}",
            )
        )
    except ImportError:
        out.append(Check("PyTorch", None, "not installed (only needed by Docling / GPU reranker)"))
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()  # fmt: skip
            out.append(Check("NVIDIA GPU", None, gpu))
        except Exception:
            pass
    return out


def run_doctor() -> int:
    results = checks()
    icon = {True: "✔", False: "✘", None: "·"}
    width = max(len(c.name) for c in results)
    for c in results:
        print(f" {icon[c.ok]} {c.name.ljust(width)}  {c.detail}")
        if c.fix and c.ok is not True:
            print(f"   {' ' * width}  → {c.fix}")
    failed = [c for c in results if c.ok is False]
    print(f"\n{len(failed)} problem(s) found." if failed else "\nAll required checks passed.")
    return 1 if failed else 0


def prefetch() -> int:
    """Download everything that is fetched lazily, so the app then runs fully offline."""
    s = get_settings()
    apply_runtime_env(s)
    status = 0
    print(f"Reranker {s.reranker_model} ...")
    from docchat.index.rerank import load_reranker

    rr = load_reranker(s)
    if rr:
        rr.score("warm up", ["model download check"])
        print("  ok")
    else:
        status = 1
        print("  failed (see warnings above)")
    print("RapidOCR models ...")
    try:
        from PIL import Image

        from docchat.ingest.ocr.rapid import RapidEngine

        RapidEngine().recognize(Image.new("RGB", (200, 60), "white"))
        print("  ok")
    except Exception as exc:
        status = 1
        print(f"  failed: {exc}")
    if backend_available("pdf", "docling") and shutil.which("docling-tools"):
        print("Docling models ...")
        subprocess.run(["docling-tools", "models", "download"], check=False)
    print("\nOllama models (run these once while online):")
    for model in dict.fromkeys([s.llm_model, s.effective_vision_model, s.embed_model]):
        print(f"  ollama pull {model}")
    print("\nThen set HF_HUB_OFFLINE=1 to guarantee no network access at runtime.")
    return status
