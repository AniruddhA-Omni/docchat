"""BM25 sparse vectors for Qdrant (IDF is applied server-side via ``Modifier.IDF``).

The tokenizer is tuned for mixed business + code corpora: thousands separators are removed
("18,450" -> "18450"), identifiers are kept whole *and* split ("INV-2025-0917", "list_items",
"camelCase"), and light plural stemming is applied. Pure Python: no model download, no ONNX.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

from qdrant_client import models

_TOKEN_RE = re.compile(r"\w+(?:[-./']\w+)*", re.UNICODE)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT_RE = re.compile(r"[-./_']")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "how",
    ]
)


def _stem(token: str) -> str:
    if len(token) > 4 and token.isalpha():
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith("s") and not token.endswith(("ss", "us", "is")):
            return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    text = _THOUSANDS_RE.sub("", text)
    out: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0)
        parts = [p for p in _SPLIT_RE.split(_CAMEL_RE.sub("_", raw)) if p]
        candidates = [raw] if len(parts) == 1 else [raw, *parts]
        for tok in candidates:
            tok = tok.lower()
            if tok in STOPWORDS or (len(tok) == 1 and not tok.isdigit()):
                continue
            out.append(_stem(tok))
    return out


def _index(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


class BM25Encoder:
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 256.0) -> None:
        self.k1, self.b, self.avg_len = k1, b, avg_len

    def encode_document(self, text: str) -> models.SparseVector:
        tokens = tokenize(text)
        counts = Counter(_index(t) for t in tokens)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_len)
        indices = list(counts)
        values = [tf * (self.k1 + 1) / (tf + norm) for tf in counts.values()]
        return models.SparseVector(indices=indices, values=values)

    def encode_query(self, text: str) -> models.SparseVector:
        indices = sorted({_index(t) for t in tokenize(text)})
        return models.SparseVector(indices=indices, values=[1.0] * len(indices))
