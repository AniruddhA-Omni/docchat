"""Confidence score for an answer, from verification results and source quality.

    confidence = 0.45 * V (claims verified) + 0.20 * R (relevance of cited sources)
               + 0.20 * C (factual sentences carrying a citation) + 0.15 * N (numbers verified)

with caps: an unverified number limits confidence to 0.5, and answers resting only on
vision-model observations are discounted (chart reading is the least reliable path).
"""

from __future__ import annotations

from docchat.agents.types import Citation, Confidence, Verdict

WEIGHTS = {"verified": 0.45, "retrieval": 0.20, "coverage": 0.20, "numeric": 0.15}


def score(verdicts: list[Verdict], citations: list[Citation]) -> Confidence:
    factual = [v for v in verdicts if v.label != "not_factual"]
    notes: list[str] = []
    if not factual:
        return Confidence(score=0.3, band="low", verified=0, retrieval=0, coverage=0, numeric=0,
                          notes=["no factual statements to verify"])  # fmt: skip
    points = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0, "uncited": 0.0}
    verified = sum(points[v.label] for v in factual) / len(factual)
    coverage = sum(1 for v in factual if v.citations) / len(factual)
    retrieval = sum(c.score for c in citations) / len(citations) if citations else 0.0
    numeric_claims = [v for v in factual if any(ch.isdigit() for ch in v.sentence)]
    numeric_bad = [v for v in numeric_claims if v.label == "unsupported" and "number" in v.reason]
    numeric = 1.0 - len(numeric_bad) / len(numeric_claims) if numeric_claims else 1.0

    value = (WEIGHTS["verified"] * verified + WEIGHTS["retrieval"] * retrieval
             + WEIGHTS["coverage"] * coverage + WEIGHTS["numeric"] * numeric)  # fmt: skip
    if numeric_bad:
        value = min(value, 0.5)
        notes.append("a number could not be verified")
    if citations and all(c.agent == "vision" for c in citations):
        value *= 0.85
        notes.append("answer relies on reading an image")
    if any(c.agent == "table" for c in citations):
        notes.append("numbers computed with SQL over the spreadsheet")
    if any(v.reason.startswith("citation") for v in factual):
        notes.append("some citations were repaired automatically")
    value = round(max(0.0, min(1.0, value)), 3)
    band = "high" if value >= 0.75 else "medium" if value >= 0.5 else "low"
    return Confidence(score=value, band=band, verified=round(verified, 3),
                      retrieval=round(retrieval, 3), coverage=round(coverage, 3),
                      numeric=round(numeric, 3), notes=notes)  # fmt: skip
