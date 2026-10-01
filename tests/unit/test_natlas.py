import pytest

from ncair_lms.errors import InvalidModelOutputError
from ncair_lms.models import Language, PortalAction, ToolName
from ncair_lms.natlas import (
    INTENT_SYSTEM_PROMPT,
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


def test_natlas_router_stages_language_intent_and_arguments():
    client = FakeClient(
        '{"language":"hausa"}',
        '{"intent":"navigation"}',
        '{"tool":"get_portal_link","arguments":{"action":"login"}}',
    )
    router = NatlasRouter(client)

    decision = router.route("Don Allah nuna min wurin shiga.")

    assert decision.language is Language.HAUSA
    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.LOGIN
    assert len(client.calls) == 3
    assert client.calls[0][0][0]["content"] == LANGUAGE_SYSTEM_PROMPT
    assert client.calls[1][0][0]["content"] == INTENT_SYSTEM_PROMPT
    assert client.calls[2][0][0]["content"].startswith(TOOL_ROUTER_SYSTEM_PROMPT)
    assert "Fixed semantic intent: navigation" in client.calls[2][0][0]["content"]
    assert "Required tool: get_portal_link" in client.calls[2][0][0]["content"]


def test_natlas_router_accepts_bare_knowledge_arguments_after_intent_stage():
    client = FakeClient(
        '{"language":"yoruba"}',
        '{"intent":"knowledge"}',
        '{"query":"course eligibility rules"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Mo fẹ mọ ofin yiyan course.")

    assert decision.language is Language.YORUBA
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "course eligibility rules"
    assert len(client.calls) == 3


def test_natlas_router_accepts_bare_step_arguments_after_intent_stage():
    client = FakeClient(
        '{"language":"english"}',
        '{"intent":"numbered_step"}',
        '{"step":2}',
    )
    router = NatlasRouter(client)

    decision = router.route("Explain onboarding step 2.")

    assert decision.tool is ToolName.STEP_GUIDANCE
    assert decision.step == 2
    assert len(client.calls) == 3


def test_natlas_router_accepts_tool_name_as_top_level_key():
    client = FakeClient(
        '{"language":"english"}',
        '{"intent":"numbered_step"}',
        '{"get_step_guidance":{"step":3}}',
    )
    router = NatlasRouter(client)

    decision = router.route("Show the third onboarding step.")

    assert decision.tool is ToolName.STEP_GUIDANCE
    assert decision.step == 3


def test_natlas_router_accepts_portal_action_shorthand():
    client = FakeClient(
        '{"language":"english"}',
        '{"intent":"navigation"}',
        '{"action":"profile"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Open my profile page.")

    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.PROFILE


def test_natlas_router_retries_invalid_intent_schema_once():
    client = FakeClient(
        '{"language":"english"}',
        '{"intent":"other"}',
        '{"intent":"still_other"}',
    )
    router = NatlasRouter(client)

    with pytest.raises(InvalidModelOutputError):
        router.route("Tell me something about my account.")

    assert len(client.calls) == 3


def test_natlas_router_does_not_allow_argument_stage_to_change_tool_family():
    client = FakeClient(
        '{"language":"english"}',
        '{"intent":"knowledge"}',
        '{"tool":"get_portal_link","arguments":{"action":"support"}}',
        '{"tool":"search_ncair_knowledge_base","arguments":{"query":"account policy"}}',
    )
    router = NatlasRouter(client)

    decision = router.route("Explain the account policy.")

    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "account policy"
    assert len(client.calls) == 4
    retry_prompt = client.calls[3][0][-1]["content"]
    assert "Keep the fixed tool search_ncair_knowledge_base" in retry_prompt


def test_natlas_router_rejects_unknown_tool_after_retry():
    client = FakeClient(
        '{"language":"english"}',
        '{"intent":"knowledge"}',
        '{"tool":"magic","arguments":{}}',
        '{"tool":"still_magic","arguments":{}}',
    )
    router = NatlasRouter(client)

    with pytest.raises(InvalidModelOutputError):
        router.route("Explain this policy.")


def test_natlas_router_repairs_single_missing_closing_brace():
    client = FakeClient(
        '{"language":"hausa"}',
        '{"intent":"knowledge"}',
        '{"tool":"search_ncair_knowledge_base","arguments":{"query":"account setup"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Ta yaya zan shirya asusun LMS?")

    assert decision.language is Language.HAUSA
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "account setup"


def test_natlas_router_still_rejects_non_truncation_json_errors():
    client = FakeClient(
        '{"language":"english"}',
        '{"intent":"knowledge"}',
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
        evidence=(
            "[guide-a.txt]\nThe first passage is related.\n\n---\n\n"
            "[guide-b.txt]\nThe second passage states the requested fact."
        ),
    )

    assert actual is expected
