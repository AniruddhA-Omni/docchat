"""Qdrant vector store with hybrid (dense + BM25) search fused by Reciprocal Rank Fusion.

Embedded mode (``QdrantClient(path=...)``) is the default: zero setup, but it takes an exclusive
lock on the folder, so only one process may open it. Set ``DOCCHAT_QDRANT_URL`` to use a Qdrant
server (``docker compose up -d qdrant``) for large corpora or when the CLI and UI run together.

Fusion is done client-side so each hit keeps its dense and sparse ranks/scores; the UI shows
them in the "How I answered" trace and the evaluator uses them for the retrieval ablation.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Literal

from qdrant_client import QdrantClient, models

from docchat.config import Settings
from docchat.schemas import Chunk

log = logging.getLogger(__name__)
SearchMode = Literal["hybrid", "dense", "sparse"]
RRF_K = 60
LOCAL_POINT_WARNING = 20_000


class StoreLockedError(RuntimeError):
    pass


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score (or raw score for single-mode search)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank: float | None = None  # cross-encoder probability, set by the reranker
    extra: dict = field(default_factory=dict)

    @property
    def relevance(self) -> float:
        """Best available relevance estimate in [0, 1]."""
        if self.rerank is not None:
            return self.rerank
        return max(0.0, self.dense_score or 0.0)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


class VectorStore:
    def __init__(self, settings: Settings, dim: int, embed_name: str) -> None:
        self.settings = settings
        self.local = not settings.qdrant_url
        self._lock = threading.RLock()
        try:
            if self.local:
                settings.qdrant_path.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(settings.qdrant_path))
            else:
                self.client = QdrantClient(url=settings.qdrant_url, timeout=60)
        except RuntimeError as exc:  # portalocker: folder already open in another process
            raise StoreLockedError(
                f"The embedded Qdrant store at {settings.qdrant_path} is in use by another process "
                "(e.g. the Streamlit app). Stop it, or run a Qdrant server and set "
                "DOCCHAT_QDRANT_URL=http://localhost:6333."
            ) from exc
        self.collection = f"{settings.qdrant_collection}__{_slug(embed_name)}__{dim}"
        self.dim = dim
        self._ensure_collection()

    def close(self) -> None:
        with self._lock:
            self.client.close()

    def _ensure_collection(self) -> None:
        with self._lock:
            if self.client.collection_exists(self.collection):
                return
            self.client.create_collection(
                self.collection,
                vectors_config={"dense": models.VectorParams(size=self.dim,
                                                             distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )  # fmt: skip
            if not self.local:
                for key in ("doc_id", "kind", "file_type"):
                    self.client.create_payload_index(self.collection, key, "keyword")
                self.client.create_payload_index(self.collection, "ordinal", "integer")

    # -- writes ----------------------------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], dense: list[list[float]],
               sparse: list[models.SparseVector]) -> None:  # fmt: skip
        points = [
            models.PointStruct(id=point_id(c.chunk_id), vector={"dense": d, "bm25": s},
                               payload=c.model_dump(mode="json"))
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]  # fmt: skip
        with self._lock:
            for i in range(0, len(points), 128):
                self.client.upsert(self.collection, points[i : i + 128], wait=True)
        if self.local and self.count() > LOCAL_POINT_WARNING:
            log.warning("embedded Qdrant holds >%d points; consider a Qdrant server",
                        LOCAL_POINT_WARNING)  # fmt: skip

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self.client.delete(self.collection, points_selector=models.FilterSelector(
                filter=_filter(doc_ids=[doc_id])), wait=True)  # fmt: skip

    def count(self, doc_id: str | None = None) -> int:
        with self._lock:
            flt = _filter(doc_ids=[doc_id]) if doc_id else None
            return self.client.count(self.collection, count_filter=flt, exact=True).count

    # -- reads -----------------------------------------------------------------------------------
    def search(
        self,
        dense: list[float] | None,
        sparse: models.SparseVector | None,
        limit: int = 30,
        doc_ids: list[str] | None = None,
        kinds: list[str] | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[Hit]:
        flt = _filter(doc_ids=doc_ids, kinds=kinds)
        dense_hits = sparse_hits = []
        with self._lock:
            if mode in ("hybrid", "dense") and dense is not None:
                dense_hits = self.client.query_points(
                    self.collection, query=dense, using="dense", limit=limit,
                    query_filter=flt, with_payload=True,
                ).points  # fmt: skip
            if mode in ("hybrid", "sparse") and sparse is not None and sparse.indices:
                sparse_hits = self.client.query_points(
                    self.collection, query=sparse, using="bm25", limit=limit,
                    query_filter=flt, with_payload=True,
                ).points  # fmt: skip
        return fuse(dense_hits, sparse_hits, limit)

    def get_chunks(self, doc_id: str, ordinals: list[int]) -> list[Chunk]:
        """Fetch specific chunks of a document (neighbour expansion)."""
        if not ordinals:
            return []
        flt = models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="ordinal", match=models.MatchAny(any=ordinals)),
        ])  # fmt: skip
        with self._lock:
            points, _ = self.client.scroll(self.collection, scroll_filter=flt,
                                           limit=len(ordinals), with_payload=True)  # fmt: skip
        return sorted((Chunk.model_validate(p.payload) for p in points), key=lambda c: c.ordinal)

    def document_chunks(self, doc_id: str, kinds: list[str] | None = None,
                        limit: int = 2000) -> list[Chunk]:  # fmt: skip
        with self._lock:
            points, _ = self.client.scroll(self.collection, limit=limit, with_payload=True,
                                           scroll_filter=_filter(doc_ids=[doc_id], kinds=kinds))  # fmt: skip
        return sorted((Chunk.model_validate(p.payload) for p in points), key=lambda c: c.ordinal)


def fuse(dense_hits, sparse_hits, limit: int) -> list[Hit]:
    """Reciprocal Rank Fusion of two ranked lists of Qdrant ScoredPoints."""
    hits: dict[str, Hit] = {}
    for rank, p in enumerate(dense_hits, 1):
        h = hits.setdefault(str(p.id), Hit(chunk=Chunk.model_validate(p.payload), score=0.0))
        h.dense_rank, h.dense_score = rank, float(p.score)
        h.score += 1.0 / (RRF_K + rank)
    for rank, p in enumerate(sparse_hits, 1):
        h = hits.setdefault(str(p.id), Hit(chunk=Chunk.model_validate(p.payload), score=0.0))
        h.sparse_rank, h.sparse_score = rank, float(p.score)
        h.score += 1.0 / (RRF_K + rank)
    return sorted(hits.values(), key=lambda h: h.score, reverse=True)[:limit]


def _filter(
    doc_ids: list[str] | None = None, kinds: list[str] | None = None
) -> models.Filter | None:
    must = []
    if doc_ids is not None:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids))))
    if kinds:
        must.append(models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds))))
    return models.Filter(must=must) if must else None
