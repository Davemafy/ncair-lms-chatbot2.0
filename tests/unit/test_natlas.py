import pytest

from ncair_lms.errors import InvalidModelOutputError
from ncair_lms.models import Language, PortalAction, ToolName
from ncair_lms.natlas import (
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


def test_language_detector_uses_its_own_model_stage():
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


def test_natlas_router_escalates_insufficient_navigation_to_knowledge():
    client = FakeClient(
        '{"language":"english"}',
        '{"tool":"get_portal_link","arguments":{"action":"courses"}}',
        '{"sufficient":false}',
        '{"query":"official course progression"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Explain the official course progression.")

    assert decision.language is Language.ENGLISH
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "official course progression"
    assert len(client.calls) == 4
    assert client.calls[3][0][0]["content"] == KNOWLEDGE_QUERY_SYSTEM_PROMPT


def test_natlas_router_escalates_insufficient_step_to_knowledge():
    client = FakeClient(
        '{"language":"english"}',
        '{"get_step_guidance":{"step":1}}',
        '{"sufficient":false}',
        '{"query":"attendance requirements for passing a cohort"}',
    )
    router = NatlasRouter(client)

    decision = router.route("What attendance do I need to pass?")

    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "attendance requirements for passing a cohort"


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


def test_natlas_router_does_not_recheck_knowledge_tool():
    client = FakeClient(
        '{"language":"yoruba"}',
        '{"action":"search_ncair_knowledge_base","query":"course eligibility rules"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Mo fẹ mọ ofin yiyan course.")

    assert decision.language is Language.YORUBA
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "course eligibility rules"
    assert len(client.calls) == 2


def test_natlas_router_accepts_knowledge_tool_name_as_top_level_key():
    client = FakeClient(
        '{"language":"english"}',
        '{"search_ncair_knowledge_base":{"query":"attendance requirement"}}',
    )
    router = NatlasRouter(client)

    decision = router.route("What attendance do I need?")

    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "attendance requirement"


def test_natlas_router_accepts_portal_action_shorthand():
    client = FakeClient(
        '{"language":"english"}',
        '{"action":"courses"}',
        '{"sufficient":true}',
    )
    router = NatlasRouter(client)

    decision = router.route("Open my courses page.")

    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.COURSES


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


def test_natlas_router_retries_invalid_knowledge_fallback_query_once():
    client = FakeClient(
        '{"language":"english"}',
        '{"action":"support"}',
        '{"sufficient":false}',
        '{"query":""}',
        '{"query":"NCAIR support policy"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Explain the NCAIR support policy.")

    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "NCAIR support policy"
    assert len(client.calls) == 5


def test_natlas_router_accepts_wrapped_knowledge_fallback_query():
    client = FakeClient(
        '{"language":"english"}',
        '{"action":"support"}',
        '{"sufficient":false}',
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


def test_natlas_router_repairs_single_missing_closing_brace():
    client = FakeClient(
        '{"language":"hausa"}',
        '{"tool":"search_ncair_knowledge_base","arguments":{"query":"NCAIR LMS account setup"}',
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


def test_tool_sufficiency_prompt_has_no_benchmark_phrases():
    assert "course progression" not in TOOL_SUFFICIENCY_SYSTEM_PROMPT.lower()
    assert "attendance" not in TOOL_SUFFICIENCY_SYSTEM_PROMPT.lower()
    assert "password" not in TOOL_SUFFICIENCY_SYSTEM_PROMPT.lower()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"supported":true}', True),
        ('{"supported":false}', False),
    ],
)
def test_evidence_verifier_returns_structured_support_verdict(raw, expected):
    verifier = NatlasEvidenceVerifier(FakeClient(raw))

    actual = verifier.is_supported(
        question="What does the guide say?",
        evidence="[guide.txt]\nThe guide states the relevant fact.",
    )

    assert actual is expected
