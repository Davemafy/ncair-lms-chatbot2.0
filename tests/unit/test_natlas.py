import pytest

from ncair_lms.errors import InvalidModelOutputError
from ncair_lms.models import Language, PortalAction, ToolName
from ncair_lms.natlas import (
    LANGUAGE_SYSTEM_PROMPT,
    TOOL_ROUTER_SYSTEM_PROMPT,
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


def test_natlas_router_separates_language_from_tool_selection():
    client = FakeClient(
        '{"language":"hausa"}',
        '{"tool":"get_portal_link","arguments":{"action":"login"}}',
    )
    router = NatlasRouter(client)

    decision = router.route("Don Allah nuna min wurin shiga.")

    assert decision.language is Language.HAUSA
    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.LOGIN
    assert len(client.calls) == 2
    assert client.calls[1][0][0]["content"] == TOOL_ROUTER_SYSTEM_PROMPT


def test_natlas_router_requires_english_retrieval_query_field():
    client = FakeClient(
        '{"language":"yoruba"}',
        '{"tool":"search_ncair_knowledge_base","arguments":{"query":"course eligibility rules"}}',
    )
    router = NatlasRouter(client)

    decision = router.route("Mo fẹ mọ ofin yiyan course.")

    assert decision.language is Language.YORUBA
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "course eligibility rules"


def test_natlas_router_retries_invalid_portal_action_with_schema():
    client = FakeClient(
        '{"language":"english"}',
        '{"tool":"get_portal_link","arguments":{"action":"dashboard_home"}}',
        '{"tool":"get_portal_link","arguments":{"action":"main"}}',
    )
    router = NatlasRouter(client)

    decision = router.route("Take me to the LMS home page.")

    assert decision.action is PortalAction.MAIN
    assert len(client.calls) == 3
    retry_prompt = client.calls[2][0][-1]["content"]
    assert "exact allowed enum values" in retry_prompt


def test_natlas_router_rejects_invalid_decision_after_retry():
    client = FakeClient(
        '{"language":"english"}',
        '{"tool":"magic","arguments":{}}',
        '{"tool":"still_magic","arguments":{}}',
    )
    router = NatlasRouter(client)

    with pytest.raises(InvalidModelOutputError):
        router.route("Help me navigate.")


def test_natlas_router_repairs_single_missing_closing_brace():
    client = FakeClient(
        '{"language":"hausa"}',
        '{"tool":"search_ncair_knowledge_base",'
        '"arguments":{"query":"NCAIR LMS account setup"}',
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
