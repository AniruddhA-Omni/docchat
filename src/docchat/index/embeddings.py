"""Dense embeddings: BGE-M3 served by Ollama (default) and a hashing embedder for tests."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from typing import Protocol

import numpy as np

from docchat.index.sparse import tokenize

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """Batched Ollama embeddings with NaN / overflow guards.

    bge-m3 on Ollama can return NaN vectors for some inputs and errors on inputs longer than its
    batch size, so each vector is checked and failures are retried one by one with shorter text.
    """

    def __init__(self, client, model: str, keep_alive: str = "30m", batch_size: int = 32,
                 max_chars: int = 6000) -> None:  # fmt: skip
        self.client, self.name = client, model
        self.keep_alive, self.batch_size, self.max_chars = keep_alive, batch_size, max_chars
        self._dim: int | None = None
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self._embed(["dimension probe"])[0])
        return self._dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embed(model=self.name, input=[t[: self.max_chars] for t in texts],
                                 truncate=True, keep_alive=self.keep_alive)  # fmt: skip
        return [list(v) for v in resp.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                vectors = self._embed(batch)
            except Exception as exc:  # one bad input fails the whole batch; isolate it
                log.warning("embedding batch failed (%s); retrying one by one", exc)
                vectors = [self._embed_one(t) for t in batch]
            out.extend(
                v if _finite(v) else self._embed_one(t) for v, t in zip(vectors, batch, strict=True)
            )
        return out

    def _embed_one(self, text: str) -> list[float]:
        for limit in (self.max_chars, 2000, 500):
            try:
                v = self._embed([text[:limit]])[0]
            except Exception as exc:
                log.warning("embedding failed at %d chars: %s", limit, exc)
                continue
            if _finite(v):
                return v
        log.error("could not embed text (%d chars); using a placeholder vector", len(text))
        return HashingEmbedder(self.dim).embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        vec = self._embed_one(text)
        self._query_cache[text] = vec
        if len(self._query_cache) > 256:
            self._query_cache.popitem(last=False)
        return vec


def _finite(v: list[float]) -> bool:
    return bool(v) and all(math.isfinite(x) for x in v) and any(x != 0 for x in v)


class HashingEmbedder:
    """Deterministic bag-of-words embedder (feature hashing). For tests and offline smoke runs."""

    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> list[float]:
        vec = np.zeros(self._dim)
        for tok in tokenize(text):
            h = hash_token(tok)
            vec[h % self._dim] += 1.0 if (h >> 20) & 1 else -1.0
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0] = 1.0
            norm = 1.0
        return (vec / norm).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def hash_token(token: str) -> int:
    import zlib

    return zlib.crc32(token.encode("utf-8"))
