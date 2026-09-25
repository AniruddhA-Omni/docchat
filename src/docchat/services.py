"""Process-wide heavy resources (catalog, vector store, embedder, reranker, LLM client).

Per-request choices that the UI can change on the fly (chat model, answer mode, parser) live in
``Settings`` and are passed alongside; ``Services`` only holds things that are expensive to build.
"""

from __future__ import annotations

import logging
import threading
from functools import cached_property

from docchat.config import Settings, apply_runtime_env
from docchat.index.catalog import Catalog
from docchat.index.embeddings import Embedder, HashingEmbedder, OllamaEmbedder
from docchat.index.rerank import Reranker, load_reranker
from docchat.index.sparse import BM25Encoder
from docchat.index.store import VectorStore
from docchat.llm import LLM, get_client

log = logging.getLogger(__name__)
_UNSET = object()


class Services:
    def __init__(self, settings: Settings, *, llm: LLM | None = None,
                 embedder: Embedder | None = None, reranker: Reranker | None | object = _UNSET,
                 store: VectorStore | None = None) -> None:  # fmt: skip
        apply_runtime_env(settings)
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = Catalog(settings.db_path, settings.data_dir / "tables")
        self.ollama = get_client(settings)
        self._llm = llm
        self._embedder = embedder
        self._reranker = reranker
        self._store = store
        self.sparse = BM25Encoder()
        self.ingest_lock = threading.Lock()
        self._init_lock = threading.Lock()

    # Heavy members are created lazily so the UI can render before models are warm.
    @property
    def llm(self) -> LLM:
        with self._init_lock:
            if self._llm is None:
                self._llm = LLM(self.ollama, self.settings)
            return self._llm

    def chat_llm(self, settings: Settings) -> LLM:
        return self.llm.with_model(settings.llm_model)

    def vision_llm(self, settings: Settings) -> LLM:
        return self.llm.with_model(settings.effective_vision_model)

    @property
    def embedder(self) -> Embedder:
        with self._init_lock:
            if self._embedder is None:
                if self.settings.embed_backend == "hashing":
                    self._embedder = HashingEmbedder()
                else:
                    self._embedder = OllamaEmbedder(self.ollama, self.settings.embed_model,
                                                    self.settings.keep_alive)  # fmt: skip
            return self._embedder

    @property
    def store(self) -> VectorStore:
        embedder = self.embedder
        with self._init_lock:
            if self._store is None:
                self._store = VectorStore(self.settings, embedder.dim, embedder.name)
            return self._store

    @property
    def reranker(self) -> Reranker | None:
        with self._init_lock:
            if self._reranker is _UNSET:
                self._reranker = load_reranker(self.settings)
            return self._reranker  # type: ignore[return-value]

    @cached_property
    def checkpointer(self):
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(
            self.settings.data_dir / "checkpoints.sqlite", check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(conn)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
        self.catalog.close()
