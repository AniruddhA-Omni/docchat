from docchat.agents.confidence import score
from docchat.agents.synthesis import is_not_found, select_evidence
from docchat.agents.types import AgentContext, Citation, Evidence, Verdict
from docchat.agents.verification import numbers, remove_unsupported, split_claims, verify
from docchat.citations.parse import best_quote, renumber


def ev(i, content, agent="retrieval", score=0.8, doc="d1", **kw):
    return Evidence(id=f"e{i}", agent=agent, kind="text", content=content, score=score, doc_id=doc,
                    file_name=f"{doc}.pdf", location=f"p.{i}", **kw)  # fmt: skip


SOURCES = [
    ev(1, "Total revenue reached USD 48.6 million in FY2025, up 18% from FY2024."),
    ev(2, "At fiscal year end the company employed 412 people across three offices."),
    ev(3, "Production deploys are frozen from December 20 to January 5."),
]


class NoLLM:
    """Context whose LLM must not be called (rules decide everything)."""

    class services:  # noqa: N801
        @staticmethod
        def chat_llm(settings):
            raise AssertionError("LLM should not be needed")

    settings = None


def test_numbers_normalization():
    assert numbers("USD 48.6 million, 18%, $1,250,000 [S2]") >= {
        "48.6",
        "48600000",
        "18",
        "1250000",
    }
    assert "2" not in numbers("see [S2]")


def test_split_claims_keeps_citations_with_sentences():
    text = "Revenue was 48.6 million. [S1] Headcount was 412 [S2].\n- Freeze in December [S3]\n| a | b |\n|---|---|"
    assert split_claims(text) == ["Revenue was 48.6 million. [S1]", "Headcount was 412 [S2].",
                                  "Freeze in December [S3]", "| a | b |"]  # fmt: skip


def test_rules_verify_supported_claims_without_llm():
    draft = (
        "Total revenue was USD 48.6 million in FY2025 [S1]. The company employed 412 people [S2]."
    )
    _, verdicts, stats = verify(NoLLM, draft, SOURCES, use_llm=True)
    assert [v.label for v in verdicts] == ["supported", "supported"]
    assert stats["llm_checked"] == 0


def test_wrong_number_is_unsupported_and_removed():
    draft = "Revenue was USD 52.1 million [S1]. The company employed 412 people [S2]."
    _, verdicts, _ = verify(NoLLM, draft, SOURCES, use_llm=False)
    assert verdicts[0].label == "unsupported" and "52.1" in verdicts[0].reason
    cleaned, removed = remove_unsupported(draft, verdicts)
    assert "52.1" not in cleaned and "412" in cleaned and len(removed) == 1


def test_citation_repair_moves_citation_to_the_right_source():
    draft = "The company employed 412 people [S1]."
    fixed, verdicts, stats = verify(NoLLM, draft, SOURCES, use_llm=False)
    assert "[S2]" in fixed and "[S1]" not in fixed
    assert verdicts[0].label == "supported" and stats["repaired_citations"] == 1


def test_uncited_claim_gets_a_citation():
    fixed, verdicts, _ = verify(NoLLM, "Production deploys are frozen from December 20 to January 5.",
                                SOURCES, use_llm=False)  # fmt: skip
    assert fixed.endswith("[S3]") and verdicts[0].citations == [3]


def test_llm_verifier_used_for_unsettled_claims():
    class Ctx:
        class services:  # noqa: N801
            @staticmethod
            def chat_llm(settings):
                from tests.fakes import FakeLLM

                return FakeLLM()

        settings = None

    draft = "Growth came mostly from new logistics customers in Asia [S1]."
    _, verdicts, stats = verify(Ctx, draft, SOURCES, use_llm=True)
    assert stats["llm_checked"] == 1 and verdicts[0].checked_by == "llm"


def test_renumber_by_first_appearance_and_drop_invalid():
    text, cites = renumber("Freeze dates [S3]. Headcount 412 [S2][S9]. Again [S3, S2].", SOURCES)
    assert text == "Freeze dates [1]. Headcount 412 [2]. Again [1][2]."
    assert [c.evidence_id for c in cites] == ["e3", "e2"]
    assert "412" in cites[1].quote


def test_best_quote_prefers_sentence_with_number():
    src = "The report covers FY2025. Headcount was 412 at year end. Offices are in three cities."
    assert best_quote("412 employees", src) == "Headcount was 412 at year end."


def test_confidence_formula_and_caps():
    good = [Verdict(sentence="a b c d 1", citations=[1], label="supported")]
    cites = [
        Citation(n=1, evidence_id="e1", agent="retrieval", file_name="f", doc_id="d", score=0.9)
    ]
    c = score(good, cites)
    assert c.band == "high" and c.score > 0.9
    bad = [*good, Verdict(sentence="x y z w 99", citations=[1], label="unsupported",
                          reason="number(s) 99 not found in the cited source")]  # fmt: skip
    assert score(bad, cites).score <= 0.5


def test_select_evidence_budget_and_specialists_first():
    items = [ev(i, f"passage {i} " * 50, score=0.5 + i / 100) for i in range(10)]
    items.append(ev(99, "RESULT: total = 5", agent="table", score=1.0))
    chosen = select_evidence(items, budget_tokens=400)
    assert chosen[0].agent == "table"
    assert sum(len(e.content) for e in chosen) / 3.2 <= 400 + 60 * len(chosen)


def test_not_found_detection():
    assert is_not_found("NOT_FOUND") and is_not_found("**NOT_FOUND**")
    assert not is_not_found("The value was found on page 2 [S1].")


def test_agent_context_type():
    assert AgentContext.__dataclass_fields__.keys() == {"services", "settings"}


HANDBOOK = ev(
    7,
    "Handover happens on Mondays at 10:00 CET. Pages must be acknowledged within 15 minutes. "
    "Deploys are not allowed on Fridays.",
)


def test_contradictions_are_never_accepted_by_rules():
    # Fuzzy similarity rates these ~90% similar to the source; rules must defer to the LLM check.
    for claim in [
        "Pages must be acknowledged within one hour [S1].",
        "Pages must never be acknowledged within 15 minutes [S1].",
        "Deploys are allowed on Fridays [S1].",
    ]:
        _, verdicts, _ = verify(NoLLM, claim, [HANDBOOK], use_llm=False)
        assert verdicts[0].label != "supported", claim
    for claim in [
        "Pages must be acknowledged within 15 minutes [S1].",
        "Deploys are not allowed on Fridays [S1].",
    ]:
        _, verdicts, _ = verify(NoLLM, claim, [HANDBOOK], use_llm=False)
        assert verdicts[0].label == "supported", claim
