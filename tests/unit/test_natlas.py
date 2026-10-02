import pytest

from ncair_lms.errors import InvalidModelOutputError
from ncair_lms.models import Language, PortalAction, ToolName
from ncair_lms.natlas import (
    NatlasLanguageDetector,
    NatlasRouter,
    OutcomeKind,
    PageLabel,
    StepLabel,
    _orthographic_language_hint,
)


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
    client = FakeClient("Igbo")
    detector = NatlasLanguageDetector(client)

    assert detector.detect("Biko nyere m aka.") is Language.IGBO
    assert len(client.choose_calls) == 1
    assert client.generate_calls == []


def test_language_detector_rejects_unknown_choice():
    client = FakeClient("Swahili")
    detector = NatlasLanguageDetector(client)

    with pytest.raises(InvalidModelOutputError):
        detector.detect("hello")


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Buɗe min shafin rajista.", Language.HAUSA),
        ("Ṣí ojú-ìwé fún mi.", Language.YORUBA),
        ("Gịnị ka m ga-eme ugbu a?", Language.IGBO),
    ],
)
def test_distinctive_orthography_can_reconcile_language(question, expected):
    client = FakeClient("English")
    detector = NatlasLanguageDetector(client)

    assert detector.detect(question) is expected


def test_orthographic_hint_ignores_non_distinctive_shared_characters():
    assert _orthographic_language_hint("Ọ bụ gị?") is None


def test_router_maps_navigation_to_canonical_portal_action():
    client = FakeClient(
        "English",
        OutcomeKind.NAVIGATION.value,
        PageLabel.SIGN_IN.value,
    )
    router = NatlasRouter(client)

    decision = router.route("Take me to the sign-in page.")

    assert decision.language is Language.ENGLISH
    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.LOGIN
    assert decision.step is None
    assert decision.retrieval_query is None
    assert client.generate_calls == []


def test_router_maps_numbered_step_deterministically():
    client = FakeClient(
        "Yoruba",
        OutcomeKind.STEP_GUIDANCE.value,
        StepLabel.STEP_4.value,
    )
    router = NatlasRouter(client)

    decision = router.route("Ṣàlàyé ìgbésẹ̀ mẹ́rin.")

    assert decision.language is Language.YORUBA
    assert decision.tool is ToolName.STEP_GUIDANCE
    assert decision.step == 4
    assert decision.action is None


def test_router_uses_original_question_for_knowledge_retrieval():
    question = "Kedu iwu banyere ndebanye aha?"
    client = FakeClient("Igbo", OutcomeKind.KNOWLEDGE.value)
    router = NatlasRouter(client)

    decision = router.route(question)

    assert decision.language is Language.IGBO
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == question
    assert decision.action is None
    assert decision.step is None
    assert len(client.choose_calls) == 2


@pytest.mark.parametrize(
    ("page", "action"),
    [
        (PageLabel.LMS_HOME, PortalAction.MAIN),
        (PageLabel.SIGN_IN, PortalAction.LOGIN),
        (PageLabel.NCAIR_HOME, PortalAction.NCAIR_HOME),
        (PageLabel.REGISTER, PortalAction.REGISTER),
        (PageLabel.PROFILE, PortalAction.PROFILE),
        (PageLabel.COURSES, PortalAction.COURSES),
        (PageLabel.TRACK_SELECTION, PortalAction.TRACK_SELECTION),
        (PageLabel.SUPPORT, PortalAction.SUPPORT),
    ],
)
def test_page_labels_have_deterministic_actions(page, action):
    client = FakeClient("English", OutcomeKind.NAVIGATION.value, page.value)
    router = NatlasRouter(client)

    decision = router.route("generic navigation request")

    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is action


@pytest.mark.parametrize(
    ("step_label", "step_number"),
    [
        (StepLabel.STEP_1, 1),
        (StepLabel.STEP_2, 2),
        (StepLabel.STEP_3, 3),
        (StepLabel.STEP_4, 4),
    ],
)
def test_step_labels_have_deterministic_numbers(step_label, step_number):
    client = FakeClient("English", OutcomeKind.STEP_GUIDANCE.value, step_label.value)
    router = NatlasRouter(client)

    decision = router.route("generic step request")

    assert decision.tool is ToolName.STEP_GUIDANCE
    assert decision.step == step_number


def test_router_rejects_unknown_outcome_choice():
    client = FakeClient("English", "something else")
    router = NatlasRouter(client)

    with pytest.raises(InvalidModelOutputError):
        router.route("generic request")


def test_router_uses_hierarchical_semantic_choice_sets():
    client = FakeClient(
        "English",
        OutcomeKind.NAVIGATION.value,
        PageLabel.COURSES.value,
    )
    router = NatlasRouter(client)

    router.route("Open my courses.")

    assert client.choose_calls[0][1] == ("English", "Hausa", "Yoruba", "Igbo")
    assert client.choose_calls[1][1] == tuple(kind.value for kind in OutcomeKind)
    assert client.choose_calls[2][1] == tuple(label.value for label in PageLabel)
