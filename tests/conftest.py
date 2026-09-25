from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def native_available(module: str, attr: str | None = None) -> bool:
    """True if a native-extension package imports (DLLs present, e.g. MSVC runtime on Windows)."""
    try:
        mod = importlib.import_module(module)
        if attr:
            getattr(mod, attr)
    except (ImportError, OSError):
        return False
    return True


requires_pymupdf = pytest.mark.skipif(not native_available("pymupdf"),
                                      reason="PyMuPDF native library not loadable")  # fmt: skip
requires_rapidocr = pytest.mark.skipif(not native_available("rapidocr", "RapidOCR"),
                                       reason="RapidOCR / onnxruntime not loadable")  # fmt: skip


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    """The generated demo corpus (built once per test session)."""
    import make_demo_corpus

    out = tmp_path_factory.mktemp("corpus")
    argv = sys.argv
    sys.argv = ["make_demo_corpus", "--out", str(out)]
    try:
        make_demo_corpus.main()
    finally:
        sys.argv = argv
    return out


@pytest.fixture
def settings(tmp_path):
    from docchat.config import Settings

    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        embed_backend="hashing",
        reranker_backend="none",
        caption_images=False,
        ollama_host="http://127.0.0.1:9",
    )


@pytest.fixture
def services(settings):
    from docchat.services import Services

    svc = Services(settings)
    yield svc
    svc.close()


@pytest.fixture
def parse_ctx(settings, tmp_path):
    from docchat.ingest.context import ParseContext

    return ParseContext(settings, "testdoc", tmp_path / "renders")
