"""Cross-encoder rerankers. Scores are returned as probabilities in [0, 1].

* ``fastembed`` (default): ONNX, CPU, ``BAAI/bge-reranker-base`` (MIT).
* ``sentence-transformers`` (extra ``rerank-gpu``): same family on CUDA, e.g. bge-reranker-v2-m3.
* ``none``: skip reranking; relevance falls back to dense cosine similarity.
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

from docchat.config import Settings

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class FastEmbedReranker:
    def __init__(self, model: str, cache_dir: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        self._model = TextCrossEncoder(model_name=model, cache_dir=cache_dir)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [_sigmoid(s) for s in self._model.rerank(query, texts, batch_size=16)]


class SentenceTransformersReranker:
    def __init__(self, model: str) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = f"{model} ({device})"
        self._model = CrossEncoder(model, device=device, max_length=512)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = self._model.predict([(query, t) for t in texts], batch_size=32).tolist()
        return [s if 0.0 <= s <= 1.0 else _sigmoid(s) for s in raw]


def load_reranker(settings: Settings) -> Reranker | None:
    backend = settings.reranker_backend
    if backend == "none":
        return None
    try:
        if backend == "sentence-transformers":
            return SentenceTransformersReranker(settings.reranker_model)
        return FastEmbedReranker(settings.reranker_model, str(settings.models_dir / "fastembed"))
    except Exception as exc:  # missing extra, DLL problems, offline without cached weights...
        log.warning("reranker %s unavailable (%s); continuing without reranking", backend, exc)
        return None
