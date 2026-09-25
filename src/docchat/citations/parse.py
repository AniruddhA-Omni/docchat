"""Citation markers: parse ``[S3]`` / ``[S1, S2]`` in model output and renumber them ``[1] [2]``
in order of first appearance, dropping ids that do not exist."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from docchat.agents.types import Citation, Evidence

CITE_RE = re.compile(r"\[\s*(S?\d+(?:\s*[,;]\s*S?\d+)*)\s*\]")
FINAL_CITE_RE = re.compile(r"\[(\d+)\]")
# Sentence boundary: after . ! ? (or after a citation that follows one), before a new sentence.
# Citations like "... January 5. [S1]" stay attached to the sentence they support.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(*\"'])|(?<=\])\s+(?=[A-Z0-9(*\"'])")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?!\[)|(?<=\])\s+(?=[A-Z0-9(*\"'])|\n+")


def cited_ids(text: str) -> list[int]:
    """Source numbers (1-based S ids) cited in ``text``, in order."""
    out = []
    for m in CITE_RE.finditer(text):
        for part in re.split(r"[,;]", m.group(1)):
            out.append(int(part.strip().lstrip("Ss")))
    return out


def strip_citations(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", CITE_RE.sub("", text)).strip()


def best_quote(claim: str, source: str, max_len: int = 320) -> str:
    """The sentence of ``source`` that best matches ``claim`` (for highlight + tooltip)."""
    candidates = [s.strip() for s in _SENT_SPLIT.split(source) if len(s.strip()) > 3]
    if not candidates:
        return source[:max_len]
    # Prefer the source sentences that contain the claim's numbers.
    nums = set(re.findall(r"\d[\d,.]*\d|\d", claim.replace(",", "")))
    with_nums = [
        c for c in candidates if nums & set(re.findall(r"\d[\d,.]*\d|\d", c.replace(",", "")))
    ]
    candidates = with_nums or candidates
    best = max(candidates, key=lambda s: fuzz.token_set_ratio(claim, s))
    return best[:max_len]


def renumber(text: str, sources: list[Evidence]) -> tuple[str, list[Citation]]:
    """Map [S#] markers to [1..n] by first appearance and build Citation objects."""
    order: dict[int, int] = {}
    claims: dict[int, list[str]] = {}

    def repl(m: re.Match) -> str:
        nums = []
        for part in re.split(r"[,;]", m.group(1)):
            sid = int(part.strip().lstrip("Ss"))
            if not 1 <= sid <= len(sources):
                continue
            if sid not in order:
                order[sid] = len(order) + 1
            nums.append(order[sid])
        return "".join(f"[{n}]" for n in dict.fromkeys(nums))

    sentences = _SENT_SPLIT.split(text)
    for sent in sentences:
        for sid in cited_ids(sent):
            claims.setdefault(sid, []).append(strip_citations(sent))
    new_text = CITE_RE.sub(repl, text)
    new_text = re.sub(r"[ \t]+(\[\d+\])", r" \1", new_text)

    citations = []
    for sid, n in sorted(order.items(), key=lambda kv: kv[1]):
        ev = sources[sid - 1]
        claim = " ".join(claims.get(sid, []))
        citations.append(Citation(
            n=n, evidence_id=ev.id, agent=ev.agent, file_name=ev.file_name, doc_id=ev.doc_id,
            file_type=ev.file_type, location=ev.location, page=ev.page, page_approx=ev.page_approx,
            bboxes=ev.bboxes, line_start=ev.line_start, line_end=ev.line_end, sheet=ev.sheet,
            table_name=ev.table_name, sql=ev.sql, row_ids=ev.row_ids, image_path=ev.image_path,
            snippet=ev.content[:1200], quote=best_quote(claim, ev.content) if claim else "",
            score=ev.score,
        ))  # fmt: skip
    return new_text, citations
