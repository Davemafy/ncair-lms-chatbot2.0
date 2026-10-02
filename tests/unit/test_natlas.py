import pytest

from ncair_lms.errors import InvalidModelOutputError
from ncair_lms.models import Language, PortalAction, ToolName
from ncair_lms.natlas import (
    NatlasEvidenceVerifier,
    NatlasLanguageDetector,
    NatlasRouter,
    _explicit_portal_action,
    _explicit_step,
    _has_navigation_intent,
    _language_hint,
)


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.generate_calls = []

    def generate(self, messages, *, max_new_tokens):
        self.generate_calls.append((messages, max_new_tokens))
        if not self.responses:
            raise AssertionError("FakeClient has no response left.")
        return self.responses.pop(0)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Where can I open the LMS?", Language.ENGLISH),
        ("Ina zan shiga LMS dina?", Language.HAUSA),
        ("Ṣí ojú ìwé LMS fún mi.", Language.YORUBA),
        ("Gịnị ka m ga-eme ugbu a?", Language.IGBO),
    ],
)
def test_language_hint_uses_general_language_signals(question, expected):
    language, margin = _language_hint(question)
    assert language is expected
    assert margin > 0


def test_language_detector_uses_model_when_hint_is_ambiguous():
    client = FakeClient('{"language":"english"}')
    detector = NatlasLanguageDetector(client)

    assert detector.detect("Explain step 4 to me.") is Language.ENGLISH
    assert len(client.generate_calls) == 1


def test_language_detector_reconciles_model_with_clear_lexical_signal():
    client = FakeClient('{"language":"english"}')
    detector = NatlasLanguageDetector(client)

    assert detector.detect("Wane irin abu nake bukata?") is Language.HAUSA


def test_language_detector_rejects_unknown_model_label():
    client = FakeClient('{"language":"swahili"}', '{"language":"swahili"}')
    detector = NatlasLanguageDetector(client)

    with pytest.raises(InvalidModelOutputError):
        detector.detect("neutral")


@pytest.mark.parametrize(
    ("question", "step"),
    [
        ("Show onboarding step 1.", 1),
        ("Me zan yi a mataki na 2?", 2),
        ("Ṣàlàyé ìgbésẹ̀ 3.", 3),
        ("Kọwaa step 4.", 4),
    ],
)
def test_explicit_step_extraction_is_structural(question, step):
    assert _explicit_step(question) == step


@pytest.mark.parametrize(
    ("question", "action"),
    [
        ("Open the LMS sign in page.", PortalAction.LOGIN),
        ("Buɗe min shafin zaɓen track.", PortalAction.TRACK_SELECTION),
        ("Ṣí ojú ìwé profile fún mi.", PortalAction.PROFILE),
        ("Meghee peeji ndebanye aha LMS.", PortalAction.REGISTER),
        ("Open NCAIR's official website.", PortalAction.NCAIR_HOME),
        ("Visit the official website of NCAIR.", PortalAction.NCAIR_HOME),
    ],
)
def test_portal_resolution_uses_destination_vocabulary(question, action):
    assert _has_navigation_intent(question)
    assert _explicit_portal_action(question) is action


def test_factual_login_question_is_not_navigation_intent():
    question = "Which credentials should a returning intern use to sign in?"
    assert not _has_navigation_intent(question)


def test_router_canonicalizes_explicit_step_even_if_model_misroutes():
    client = FakeClient(
        '{"language":"english"}',
        '{"tool":"get_portal_link"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Explain step 3 to me.")

    assert decision.tool is ToolName.STEP_GUIDANCE
    assert decision.step == 3


def test_router_canonicalizes_explicit_navigation_even_if_model_misroutes():
    client = FakeClient(
        '{"language":"english"}',
        '{"tool":"search_ncair_knowledge_base"}',
    )
    router = NatlasRouter(client)

    decision = router.route("Open the courses page for me.")

    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.COURSES


def test_router_rejects_specialized_tool_without_contract_precondition():
    client = FakeClient(
        '{"language":"english"}',
        '{"tool":"get_portal_link"}',
    )
    router = NatlasRouter(client)

    decision = router.route("What are the registration requirements?")

    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "What are the registration requirements?"


def test_router_keeps_original_multilingual_question_for_knowledge():
    client = FakeClient(
        '{"language":"igbo"}',
        '{"tool":"search_ncair_knowledge_base"}',
    )
    router = NatlasRouter(client)

    question = "Kedu iwu maka password LMS?"
    decision = router.route(question)

    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == question


def test_router_retries_invalid_tool_json():
    client = FakeClient(
        '{"tool":"not_real"}',
        '{"tool":"search_ncair_knowledge_base"}',
    )
    router = NatlasRouter(client)

    decision = router.route("What is the attendance rule?")

    assert decision.tool is ToolName.KNOWLEDGE
    assert len(client.generate_calls) == 2


def test_evidence_verifier_accepts_supported_passage_index():
    evidence = (
        "[ncair_knowledge_base.txt]\n"
        "Attendance Threshold: A minimum of 75% attendance is required."
    )
    client = FakeClient(
        '{"verdict":"supported","passage_index":1}'
    )
    verifier = NatlasEvidenceVerifier(client)

    assert verifier.is_supported(question="What attendance is required?", evidence=evidence) is True


def test_evidence_verifier_accepts_contradicted_passage_index():
    evidence = (
        "[ncair_knowledge_base.txt]\n"
        "Returning interns do NOT need to onboard again. "
        "They should use their existing email and password."
    )
    client = FakeClient(
        '{"verdict":"contradicted","passage_index":1}'
    )
    verifier = NatlasEvidenceVerifier(client)

    assert (
        verifier.is_supported(
            question="Returning interns must onboard again, right?",
            evidence=evidence,
        )
        is True
    )


def test_evidence_verifier_rejects_not_found_with_null_index():
    client = FakeClient('{"verdict":"not_found","passage_index":null}')
    verifier = NatlasEvidenceVerifier(client)

    assert (
        verifier.is_supported(
            question="What stipend is paid?",
            evidence="[ncair_knowledge_base.txt]\nAttendance must be at least 75%.",
        )
        is False
    )


def test_evidence_verifier_rejects_out_of_range_index_after_retry():
    evidence = (
        "[ncair_knowledge_base.txt]\nAttendance must be at least 75%.\n\n"
        "---\n\n"
        "[ncair_knowledge_base.txt]\nRegistration closes at 5:00 PM."
    )
    client = FakeClient(
        '{"verdict":"supported","passage_index":7}',
        '{"verdict":"supported","passage_index":9}',
    )
    verifier = NatlasEvidenceVerifier(client)

    assert verifier.is_supported(question="What stipend is paid?", evidence=evidence) is False
    assert len(client.generate_calls) == 2


def test_evidence_verifier_retries_invalid_shape():
    evidence = (
        "[ncair_knowledge_base.txt]\nRegistration closes at 5:00 PM."
    )
    client = FakeClient(
        '{"supported":true}',
        '{"verdict":"supported","passage_index":1}',
    )
    verifier = NatlasEvidenceVerifier(client)

    assert (
        verifier.is_supported(
            question="When does registration close?",
            evidence=evidence,
        )
        is True
    )
    assert len(client.generate_calls) == 2


def test_evidence_verifier_rejects_not_found_with_non_null_index():
    evidence = "[ncair_knowledge_base.txt]\nAttendance must be at least 75%."
    client = FakeClient(
        '{"verdict":"not_found","passage_index":1}',
        '{"verdict":"not_found","passage_index":1}',
    )
    verifier = NatlasEvidenceVerifier(client)

    assert verifier.is_supported(question="What stipend is paid?", evidence=evidence) is False
    assert len(client.generate_calls) == 2
