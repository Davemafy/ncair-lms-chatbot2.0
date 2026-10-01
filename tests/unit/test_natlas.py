import pytest

from ncair_lms.errors import InvalidModelOutputError
from ncair_lms.models import Language, PortalAction, ToolName
from ncair_lms.natlas import NatlasRouter


class FakeClient:
    def __init__(self, output):
        self.output = output
        self.messages = None

    def generate(self, messages, *, max_new_tokens):
        self.messages = messages
        del max_new_tokens
        return self.output


def test_natlas_router_accepts_multilingual_structured_decision():
    router = NatlasRouter(
        FakeClient('{"language":"hausa","tool":"get_portal_link","arguments":{"action":"login"}}'),
    )
    decision = router.route("Ina zan shiga LMS?")
    assert decision.language is Language.HAUSA
    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.LOGIN


def test_natlas_router_requires_english_retrieval_query_field():
    router = NatlasRouter(
        FakeClient(
            '{"language":"yoruba","tool":"search_ncair_knowledge_base",'
            '"arguments":{"query":"attendance requirement"}}'
        ),
    )
    decision = router.route("Attendance mélòó ni mo nilo?")
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "attendance requirement"


def test_natlas_router_rejects_unknown_tool():
    router = NatlasRouter(
        FakeClient('{"language":"english","tool":"magic","arguments":{}}'),
    )
    with pytest.raises(InvalidModelOutputError):
        router.route("hello")


def test_router_prompt_keeps_user_language_and_reserves_step_tool_for_numbered_steps():
    client = FakeClient(
        '{"language":"hausa","tool":"search_ncair_knowledge_base",'
        '"arguments":{"query":"NCAIR LMS onboarding getting started"}}'
    )
    router = NatlasRouter(client)

    decision = router.route("Don fara amfani da NCAIR LMS, me zan yi?")

    assert decision.language is Language.HAUSA
    assert decision.tool is ToolName.KNOWLEDGE
    assert client.messages is not None
    system_prompt = client.messages[0]["content"]
    assert '"language" field must identify the language of the user\'s input' in system_prompt
    expected = "Use get_step_guidance only when the user explicitly refers to a numbered"
    assert expected in system_prompt
    assert "Don fara amfani da NCAIR LMS, me zan yi?" in system_prompt
    assert '"language":"hausa","tool":"search_ncair_knowledge_base"' in system_prompt
