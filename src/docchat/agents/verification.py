"""Citation / Verification agent.

1. Deterministic checks (every mode): each factual sentence must cite a source that exists, and
   every number in it must appear in a cited source. Wrong or missing citations are repaired
   when another source contains the claim ("citation repair").
2. One batched LLM check (balanced / deep) for sentences the rules could not settle, using only
   the best-matching excerpts of the cited sources.
3. Deep mode: one LLM revision pass rewrites the flagged sentences, then the rules run again.
4. Whatever is still unsupported is removed from the answer.
"""

from __future__ import annotations

import re

from pydantic import BaseModel
from rapidfuzz import fuzz

from docchat.agents.types import AgentContext, Evidence, Verdict
from docchat.citations.parse import CITE_RE, best_quote, cited_ids, strip_citations
from docchat.citations.parse import SENTENCE_SPLIT as _SPLIT
from docchat.index.sparse import tokenize
from docchat.llm import LLMError
from docchat.llm.prompts import REVISE_SYSTEM, VERIFY_SYSTEM

_NUM_RE = re.compile(
    r"(?<![\w.])[-+]?[$€£¥₹]?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|million|billion|thousand|[kKmMbB]\b)?"
)
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9}
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}")
# Rules accept a claim only when (almost) every key term of it appears in the cited source.
# Fuzzy similarity is not used for acceptance: "within one hour" vs "within 15 minutes" or an
# added "never" still look ~90% similar. Anything below these thresholds goes to the LLM check.
RULE_COVERAGE = 0.9
RULE_COVERAGE_NUMERIC = 0.75  # claims whose numbers were all verified in the source


_NEGATION = re.compile(
    r"\b(not|never|no|none|neither|nor|without|cannot|nobody|nothing)\b|n't\b", re.I
)


def negation_mismatch(claim: str, source_sentence: str) -> bool:
    """True if exactly one of claim / best-matching source sentence is negated."""
    return bool(_NEGATION.search(claim)) != bool(_NEGATION.search(source_sentence))


def term_coverage(claim: str, source: str) -> float:
    """Share of the claim's key terms (BM25 tokens: no stop words, light stemming) in the source.
    Negations such as 'not' / 'never' are key terms, so they are never ignored."""
    terms = set(tokenize(claim))
    if not terms:
        return 0.0
    return len(terms & set(tokenize(source))) / len(terms)


class _LLMVerdict(BaseModel):
    i: int
    label: str


class _LLMVerdicts(BaseModel):
    verdicts: list[_LLMVerdict]


def split_claims(answer: str) -> list[str]:
    """Split an answer into checkable units: sentences, bullet items and table rows."""
    units = []
    for line in answer.split("\n"):
        line = line.strip()
        if not line or _TABLE_RULE.match(line):
            continue
        if line.startswith("|"):
            units.append(line)
            continue
        line = _LIST_MARK.sub("", line)
        units.extend(s.strip() for s in _SPLIT.split(line) if s.strip())
    return units


def numbers(text: str) -> set[str]:
    """Normalized numbers in ``text`` (raw and scaled forms): '1.25 million' -> {'1.25','1250000'}."""
    out = set()
    for m in _NUM_RE.finditer(strip_citations(text)):
        raw = m.group(0).strip()
        unit = re.search(r"(%|percent|million|billion|thousand|[kKmMbB])$", raw)
        digits = re.sub(r"[^\d.\-]", "", raw.replace(",", ""))
        if not digits or digits in "-.":
            continue
        try:
            value = float(digits)
        except ValueError:
            continue
        out.add(_canon(value))
        if unit and unit.group(1).lower() in _SCALE:
            out.add(_canon(value * _SCALE[unit.group(1).lower()]))
    return out


def _canon(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.6g}"


def _is_factual(claim: str) -> bool:
    text = strip_citations(claim)
    if len(text.split()) < 4 and not numbers(text):
        return False
    return not text.endswith(":")


def verify(ctx: AgentContext, draft: str, sources: list[Evidence],
           use_llm: bool = True) -> tuple[str, list[Verdict], dict]:  # fmt: skip
    """Return (possibly citation-repaired draft, verdicts, stats)."""
    source_numbers = [numbers(s.content) for s in sources]
    verdicts: list[Verdict] = []
    pending: list[tuple[int, str, list[int]]] = []
    repaired = 0
    new_draft = draft

    for claim in split_claims(draft):
        cites = [c for c in cited_ids(claim) if 1 <= c <= len(sources)]
        plain = strip_citations(claim)
        v = Verdict(sentence=plain, citations=cites)
        if not _is_factual(claim):
            v.label = "not_factual"
            verdicts.append(v)
            continue
        claim_nums = {n for n in numbers(plain) if not _is_year_or_index(n, plain)}
        if not cites:
            fix = _find_source(plain, claim_nums, sources, source_numbers)
            if fix is None:
                v.label, v.reason = "uncited", "no citation and no source matches"
                verdicts.append(v)
                continue
            cites = v.citations = [fix]
            new_draft = new_draft.replace(claim, f"{claim.rstrip()} [S{fix}]", 1)
            repaired += 1
            v.reason = "citation added"
        missing = claim_nums - set().union(*(source_numbers[c - 1] for c in cites))
        if missing:
            fix = _find_source(plain, claim_nums, sources, source_numbers)
            if fix is not None and fix not in cites:
                new_claim = CITE_RE.sub("", claim).rstrip() + f" [S{fix}]"
                new_draft = new_draft.replace(claim, new_claim, 1)
                cites = v.citations = [fix]
                repaired += 1
                v.reason = "citation corrected"
                missing = claim_nums - source_numbers[fix - 1]
        if missing:
            v.label = "unsupported"
            v.reason = f"number(s) {', '.join(sorted(missing))} not found in the cited source"
            verdicts.append(v)
            continue
        cited_text = " ".join(sources[c - 1].content for c in cites)
        cover = term_coverage(plain, cited_text)
        negated = negation_mismatch(plain, best_quote(plain, cited_text))
        if not negated and (
            cover >= RULE_COVERAGE or (claim_nums and cover >= RULE_COVERAGE_NUMERIC)
        ):
            v.label = "supported"
            v.reason = v.reason or f"{cover:.0%} of its key terms are in the source"
            verdicts.append(v)
            continue
        verdicts.append(v)
        pending.append((len(verdicts) - 1, plain, cites))

    if pending and use_llm:
        _llm_check(ctx, pending, sources, verdicts)
    else:
        for idx, _, _ in pending:
            verdicts[idx].label = "partial"
            verdicts[idx].reason = "not checked (fast mode)"

    stats = {
        "claims": len(verdicts),
        "repaired_citations": repaired,
        "llm_checked": len(pending) if use_llm else 0,
    }
    return new_draft, verdicts, stats


def _is_year_or_index(n: str, text: str) -> bool:
    return len(n) == 4 and n.startswith(("19", "20")) and n in text


def _find_source(claim: str, claim_nums: set[str], sources: list[Evidence],
                 source_numbers: list[set[str]]) -> int | None:  # fmt: skip
    best, best_score = None, 0.0
    for i, (src, nums) in enumerate(zip(sources, source_numbers, strict=True), 1):
        if claim_nums and not claim_nums <= nums:
            continue
        score = fuzz.token_set_ratio(claim, best_quote(claim, src.content))
        if score > best_score:
            best, best_score = i, score
    return best if best_score >= 70 else None


def _llm_check(
    ctx: AgentContext, pending, sources: list[Evidence], verdicts: list[Verdict]
) -> None:
    lines = []
    for k, (_, claim, cites) in enumerate(pending, 1):
        excerpts = " | ".join(best_quote(claim, sources[c - 1].content, 400) for c in cites[:2])
        lines.append(f"Claim {k}: {claim}\nExcerpt {k}: {excerpts}")
    llm = ctx.services.chat_llm(ctx.settings)
    try:
        out = llm.json([{"role": "user", "content": "\n\n".join(lines)}], _LLMVerdicts,
                       system=VERIFY_SYSTEM, tag="verify", num_predict=60 + 25 * len(pending))  # fmt: skip
        labels = {v.i: v.label.lower().strip() for v in out.verdicts}
    except LLMError:
        labels = {}
    for k, (idx, _, _) in enumerate(pending, 1):
        label = labels.get(k)
        if label in ("supported", "partial", "unsupported"):
            verdicts[idx].label = label
            verdicts[idx].checked_by = "llm"
        else:
            verdicts[idx].label = "partial"
            verdicts[idx].reason = "verifier gave no verdict"


def revise(ctx: AgentContext, question: str, draft: str, flagged: list[Verdict],
           sources: list[Evidence]) -> str | None:  # fmt: skip
    """Ask the model to fix the flagged sentences (deep mode). Returns None if it gives up."""
    from docchat.agents.synthesis import format_sources

    issues = "\n".join(f"- {v.sentence} ({v.reason or 'not supported by the cited source'})"
                       for v in flagged)  # fmt: skip
    prompt = (f"Question: {question}\n\nSources:\n{format_sources(sources)}\n\n"
              f"Draft answer:\n{draft}\n\nFlagged statements:\n{issues}\n\n"
              "Rewrite the draft answer.")  # fmt: skip
    try:
        res = ctx.services.chat_llm(ctx.settings).chat(
            [{"role": "user", "content": prompt}], system=REVISE_SYSTEM, tag="revise",
            think=False, temperature=0.0, num_predict=1024,
        )  # fmt: skip
    except LLMError:
        return None
    text = res.content.strip()
    if not text or text.upper().startswith("NOT_FOUND"):
        return None
    return text


def remove_unsupported(draft: str, verdicts: list[Verdict]) -> tuple[str, list[str]]:
    """Drop unsupported sentences; keep the rest verbatim."""
    bad = [v.sentence for v in verdicts if v.label == "unsupported"]
    if not bad:
        return draft, []
    kept_lines = []
    for line in draft.split("\n"):
        pieces = _SPLIT.split(line) if not line.strip().startswith("|") else [line]
        kept = [p for p in pieces if strip_citations(_LIST_MARK.sub("", p.strip())) not in bad]
        if kept or not line.strip():
            prefix = _LIST_MARK.match(line)
            text = " ".join(kept)
            if prefix and kept and not _LIST_MARK.match(text):
                text = prefix.group(0) + text
            kept_lines.append(text)
    return "\n".join(kept_lines).strip(), bad
