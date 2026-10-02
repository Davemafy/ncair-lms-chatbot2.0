import pytest

from ncair_lms.errors import InvalidModelOutputError
from ncair_lms.models import Language, PortalAction, ToolName
from ncair_lms.natlas import (
    KNOWLEDGE_CHALLENGE_SYSTEM_PROMPT,
    KNOWLEDGE_QUERY_SYSTEM_PROMPT,
    LANGUAGE_SYSTEM_PROMPT,
    TOOL_ROUTER_SYSTEM_PROMPT,
    TOOL_SUFFICIENCY_SYSTEM_PROMPT,
    NatlasEvidenceVerifier,
    NatlasLanguageDetector,
    NatlasRouter,
)


class FakeClient:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, messages, *, max_new_tokens):
        self.calls.append((messages, max_new_tokens))
        if not self.outputs:
            raise AssertionError("FakeClient has no output left.")
        return self.outputs.pop(0)


def test_language_detector_uses_model_when_orthography_is_not_decisive():
    client = FakeClient('{"language":"igbo"}')
    detector = NatlasLanguageDetector(client)

    language = detector.detect("Biko nyere m aka.")

    assert language is Language.IGBO
    assert len(client.calls) == 1
    assert client.calls[0][0][0]["content"] == LANGUAGE_SYSTEM_PROMPT


def test_language_detector_retries_invalid_schema_once():
    client = FakeClient('{"language":"unknown"}', '{"language":"yoruba"}')
    detector = NatlasLanguageDetector(client)

    assert detector.detect("Jọwọ ran mi lọwọ.") is Language.YORUBA
    assert len(client.calls) == 2


def test_language_detector_orthography_overrides_conflicting_yoruba_label():
    client = FakeClient('{"language":"hausa"}')
    detector = NatlasLanguageDetector(client)

    assert detector.detect("Ṣé ẹ lè ṣí ojú-ìwé náà?") is Language.YORUBA
    assert len(client.calls) == 1


def test_language_detector_orthography_overrides_conflicting_igbo_label():
    client = FakeClient('{"language":"english"}')
    detector = NatlasLanguageDetector(client)

    assert detector.detect("Biko kọwaa ihe dị n'ime akwụkwọ ahụ.") is Language.IGBO
    assert len(client.calls) == 1


def test_language_detector_orthography_overrides_conflicting_hausa_label():
    client = FakeClient('{"language":"english"}')
    detector = NatlasLanguageDetector(client)

    assert detector.detect("Don Allah ƙara bayani ɗaya.") is Language.HAUSA
    assert len(client.calls) == 1


def test_natlas_router_keeps_sufficient_navigation_decision():
    client = FakeClient(
        '{"language":"hausa"}',
        '{"tool":"get_portal_link","arguments":{"action":"login"}}',
        '{"sufficient":true}',
    )
    router = NatlasRouter(client)

    decision = router.route("Don Allah nuna min wurin shiga.")

    assert decision.language is Language.HAUSA
    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.LOGIN
    assert len(client.calls) == 3
    assert client.calls[1][0][0]["content"] == TOOL_ROUTER_SYSTEM_PROMPT
    assert client.calls[2][0][0]["content"] == TOOL_SUFFICIENCY_SYSTEM_PROMPT


def test_natlas_router_requires_consensus_before_escalating_navigation():
    client = FakeClient(
        '{"language":"english"}',
        '{"tool":"get_portal_link","arguments":{"action":"courses"}}',
        '{"sufficient":false}',
        '{"knowledge_required":false}',
    )
    router = NatlasRouter(client)

    decision = router.route("Take me to the courses destination.")

    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.COURSES
    assert len(client.calls) == 4
    assert client.calls[3][0][0]["content"] == KNOWLEDGE_CHALLENGE_SYSTEM_PROMPT


def test_natlas_router_escalates_only_when_challenger_confirms_knowledge():
    client = FakeClient(
        '{"language":"english"}',
        '{"tool":"get_portal_link","arguments":{"action":"courses"}}',
        '{"sufficient":false}',
        '{"knowledge_required":true}',
        '{"query":"official course progression rules"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Explain the official progression rules.")

    assert decision.language is Language.ENGLISH
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "official course progression rules"
    assert len(client.calls) == 5
    assert client.calls[4][0][0]["content"] == KNOWLEDGE_QUERY_SYSTEM_PROMPT


def test_natlas_router_escalates_step_only_after_confirmation():
    client = FakeClient(
        '{"language":"english"}',
        '{"get_step_guidance":{"step":1}}',
        '{"sufficient":false}',
        '{"knowledge_required":true}',
        '{"query":"cohort attendance requirement"}',
    )
    router = NatlasRouter(client)

    decision = router.route("What attendance is required to pass a cohort?")

    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "cohort attendance requirement"


def test_natlas_router_keeps_explicit_step_when_sufficient():
    client = FakeClient(
        '{"language":"english"}',
        '{"get_step_guidance":{"step":2}}',
        '{"sufficient":true}',
    )
    router = NatlasRouter(client)

    decision = router.route("Show me onboarding step 2.")

    assert decision.tool is ToolName.STEP_GUIDANCE
    assert decision.step == 2


def test_natlas_router_canonicalizes_direct_knowledge_query():
    client = FakeClient(
        '{"language":"yoruba"}',
        '{"action":"search_ncair_knowledge_base","query":"rough mixed-language query"}',
        '{"query":"course eligibility requirements"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Mo fẹ mọ ofin yiyan course.")

    assert decision.language is Language.YORUBA
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "course eligibility requirements"
    assert len(client.calls) == 3
    assert client.calls[2][0][0]["content"] == KNOWLEDGE_QUERY_SYSTEM_PROMPT


def test_natlas_router_uses_router_query_if_canonicalizer_fails_twice():
    client = FakeClient(
        '{"language":"english"}',
        '{"search_ncair_knowledge_base":{"query":"account requirement"}}',
        '{"query":""}',
        '{"query":""}',
    )
    router = NatlasRouter(client)

    decision = router.route("What does the official account policy require?")

    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "account requirement"
    assert len(client.calls) == 4


def test_natlas_router_accepts_portal_action_shorthand():
    client = FakeClient(
        '{"language":"english"}',
        '{"action":"profile"}',
        '{"sufficient":true}',
    )
    router = NatlasRouter(client)

    decision = router.route("Open my profile page.")

    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.PROFILE


def test_natlas_router_retries_invalid_portal_action_then_checks_sufficiency():
    client = FakeClient(
        '{"language":"english"}',
        '{"tool":"get_portal_link","arguments":{"action":"dashboard_home"}}',
        '{"tool":"get_portal_link","arguments":{"action":"main"}}',
        '{"sufficient":true}',
    )
    router = NatlasRouter(client)

    decision = router.route("Take me to the LMS home page.")

    assert decision.action is PortalAction.MAIN
    assert len(client.calls) == 4
    retry_prompt = client.calls[2][0][-1]["content"]
    assert "exact allowed enum values" in retry_prompt


def test_natlas_router_retries_invalid_sufficiency_verdict_once():
    client = FakeClient(
        '{"language":"english"}',
        '{"action":"profile"}',
        '{"supported":true}',
        '{"sufficient":true}',
    )
    router = NatlasRouter(client)

    decision = router.route("Open my profile page.")

    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.PROFILE
    assert len(client.calls) == 4


def test_natlas_router_retries_invalid_challenge_verdict_once():
    client = FakeClient(
        '{"language":"english"}',
        '{"action":"support"}',
        '{"sufficient":false}',
        '{"required":true}',
        '{"knowledge_required":false}',
    )
    router = NatlasRouter(client)

    decision = router.route("Take me to the support destination.")

    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.SUPPORT
    assert len(client.calls) == 5


def test_natlas_router_retries_invalid_knowledge_fallback_query_once():
    client = FakeClient(
        '{"language":"english"}',
        '{"action":"support"}',
        '{"sufficient":false}',
        '{"knowledge_required":true}',
        '{"query":""}',
        '{"query":"NCAIR support policy"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Explain the NCAIR support policy.")

    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "NCAIR support policy"
    assert len(client.calls) == 6


def test_natlas_router_accepts_wrapped_knowledge_fallback_query():
    client = FakeClient(
        '{"language":"english"}',
        '{"action":"support"}',
        '{"sufficient":false}',
        '{"knowledge_required":true}',
        '{"search_ncair_knowledge_base":{"query":"support requirements"}}',
    )
    router = NatlasRouter(client)

    decision = router.route("Explain the support requirements.")

    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "support requirements"


def test_natlas_router_rejects_unknown_flat_action_after_retry():
    client = FakeClient(
        '{"language":"english"}',
        '{"action":"invented_tool","query":"x"}',
        '{"action":"still_invented","query":"x"}',
    )
    router = NatlasRouter(client)

    with pytest.raises(InvalidModelOutputError):
        router.route("Find this information.")


def test_natlas_router_repairs_single_missing_closing_brace_then_canonicalizes_query():
    client = FakeClient(
        '{"language":"hausa"}',
        '{"tool":"search_ncair_knowledge_base","arguments":{"query":"account setup"}',
        '{"query":"NCAIR LMS account setup"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Ta yaya zan shirya asusun LMS?")

    assert decision.language is Language.HAUSA
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "NCAIR LMS account setup"


def test_natlas_router_still_rejects_non_truncation_json_errors():
    client = FakeClient(
        '{"language":"english"}',
        '{"tool":search_ncair_knowledge_base,"arguments":{"query":"onboarding"}}',
        '{"tool":search_ncair_knowledge_base,"arguments":{"query":"onboarding"}}',
    )
    router = NatlasRouter(client)

    with pytest.raises(InvalidModelOutputError):
        router.route("Explain onboarding.")


def test_guard_prompts_are_generic_not_benchmark_specific():
    combined = (
        TOOL_SUFFICIENCY_SYSTEM_PROMPT
        + KNOWLEDGE_CHALLENGE_SYSTEM_PROMPT
        + KNOWLEDGE_QUERY_SYSTEM_PROMPT
    ).lower()
    assert "course progression" not in combined
    assert "attendance" not in combined
    assert "password" not in combined
    assert "logbook" not in combined


def test_evidence_verifier_accepts_exact_supporting_quote():
    verifier = NatlasEvidenceVerifier(
        FakeClient('{"supported":true,"quote":"requires 75% attendance"}')
    )

    actual = verifier.is_supported(
        question="What does the guide require?",
        evidence="[guide.txt]\nThe cohort requires 75% attendance to pass.",
    )

    assert actual is True


def test_evidence_verifier_rejects_unsupported_passage():
    verifier = NatlasEvidenceVerifier(FakeClient('{"supported":false,"quote":""}'))

    actual = verifier.is_supported(
        question="What stipend is paid?",
        evidence="[guide.txt]\nThe guide explains course attendance.",
    )

    assert actual is False


def test_evidence_verifier_retries_when_quote_is_not_extractive():
    verifier = NatlasEvidenceVerifier(
        FakeClient(
            '{"supported":true,"quote":"a paraphrase not in the evidence"}',
            '{"supported":true,"quote":"PDF or PNG under 2 MB"}',
        )
    )

    actual = verifier.is_supported(
        question="Which upload formats are allowed?",
        evidence="[guide.txt]\nUpload documents as PDF or PNG under 2 MB.",
    )

    assert actual is True
    assert len(verifier._client.calls) == 2
