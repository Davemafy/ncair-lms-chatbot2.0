import pytest

from ncair_lms.errors import InvalidModelOutputError
from ncair_lms.models import Language, PortalAction, ToolName
from ncair_lms.natlas import NatlasLanguageDetector, NatlasRouter, RouteLabel


class FakeClient:
    def __init__(self, *choices):
        self.choices = list(choices)
        self.choose_calls = []
        self.generate_calls = []

    def choose(self, messages, choices):
        self.choose_calls.append((messages, tuple(choices)))
        if not self.choices:
            raise AssertionError("FakeClient has no classification choice left.")
        return self.choices.pop(0)

    def generate(self, messages, *, max_new_tokens):
        self.generate_calls.append((messages, max_new_tokens))
        return "grounded answer"


def test_language_detector_maps_closed_choice_without_generation():
    client = FakeClient("igbo")
    detector = NatlasLanguageDetector(client)

    assert detector.detect("Biko nyere m aka.") is Language.IGBO
    assert len(client.choose_calls) == 1
    assert client.generate_calls == []


def test_language_detector_rejects_unknown_choice():
    client = FakeClient("swahili")
    detector = NatlasLanguageDetector(client)

    with pytest.raises(InvalidModelOutputError):
        detector.detect("hello")


def test_router_maps_signin_class_to_canonical_portal_action():
    client = FakeClient("english", RouteLabel.SIGN_IN.value)
    router = NatlasRouter(client)

    decision = router.route("Take me to the sign-in page.")

    assert decision.language is Language.ENGLISH
    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.LOGIN
    assert decision.step is None
    assert decision.retrieval_query is None
    assert client.generate_calls == []


def test_router_maps_numbered_step_class_deterministically():
    client = FakeClient("yoruba", RouteLabel.STEP_4.value)
    router = NatlasRouter(client)

    decision = router.route("Ṣàlàyé ìgbésẹ̀ mẹ́rin.")

    assert decision.language is Language.YORUBA
    assert decision.tool is ToolName.STEP_GUIDANCE
    assert decision.step == 4
    assert decision.action is None


def test_router_uses_original_multilingual_question_for_knowledge_retrieval():
    question = "Kedu iwu banyere ndebanye aha?"
    client = FakeClient("igbo", RouteLabel.KNOWLEDGE.value)
    router = NatlasRouter(client)

    decision = router.route(question)

    assert decision.language is Language.IGBO
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == question
    assert decision.action is None
    assert decision.step is None


@pytest.mark.parametrize(
    ("label", "tool", "action", "step"),
    [
        (RouteLabel.LMS_HOME, ToolName.PORTAL_LINK, PortalAction.MAIN, None),
        (RouteLabel.NCAIR_HOME, ToolName.PORTAL_LINK, PortalAction.NCAIR_HOME, None),
        (RouteLabel.REGISTER, ToolName.PORTAL_LINK, PortalAction.REGISTER, None),
        (RouteLabel.PROFILE, ToolName.PORTAL_LINK, PortalAction.PROFILE, None),
        (RouteLabel.COURSES, ToolName.PORTAL_LINK, PortalAction.COURSES, None),
        (
            RouteLabel.TRACK_SELECTION,
            ToolName.PORTAL_LINK,
            PortalAction.TRACK_SELECTION,
            None,
        ),
        (RouteLabel.SUPPORT, ToolName.PORTAL_LINK, PortalAction.SUPPORT, None),
        (RouteLabel.STEP_1, ToolName.STEP_GUIDANCE, None, 1),
        (RouteLabel.STEP_2, ToolName.STEP_GUIDANCE, None, 2),
        (RouteLabel.STEP_3, ToolName.STEP_GUIDANCE, None, 3),
    ],
)
def test_route_labels_have_deterministic_arguments(label, tool, action, step):
    client = FakeClient("hausa", label.value)
    router = NatlasRouter(client)

    decision = router.route("generic request")

    assert decision.language is Language.HAUSA
    assert decision.tool is tool
    assert decision.action is action
    assert decision.step == step


def test_router_rejects_unknown_route_choice():
    client = FakeClient("english", "not-a-route")
    router = NatlasRouter(client)

    with pytest.raises(InvalidModelOutputError):
        router.route("generic request")


def test_router_uses_exact_closed_choice_sets():
    client = FakeClient("english", RouteLabel.KNOWLEDGE.value)
    router = NatlasRouter(client)

    router.route("Explain the policy.")

    language_choices = client.choose_calls[0][1]
    route_choices = client.choose_calls[1][1]
    assert language_choices == ("english", "hausa", "yoruba", "igbo")
    assert route_choices == tuple(label.value for label in RouteLabel)


def test_route_labels_are_semantic_not_opaque_codes():
    assert RouteLabel.LMS_HOME.value == "home"
    assert RouteLabel.SIGN_IN.value == "login"
    assert RouteLabel.STEP_4.value == "fourth"
    assert RouteLabel.KNOWLEDGE.value == "knowledge"
    assert all(len(label.value) > 1 for label in RouteLabel)
