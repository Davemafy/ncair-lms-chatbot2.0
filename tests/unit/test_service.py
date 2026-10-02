from ncair_lms.config import Settings
from ncair_lms.models import (
    EvidencePassage,
    Language,
    RetrievedEvidence,
    RoutingDecision,
    ToolName,
)
from ncair_lms.service import AssistantService


class FakeKnowledgeBase:
    def search(self, query, *, top_k):
        del query, top_k
        return RetrievedEvidence(
            (
                EvidencePassage(source="first.txt", text="Related but insufficient information."),
                EvidencePassage(source="guide.txt", text="Attendance requires 75%."),
                EvidencePassage(source="other.txt", text="Unrelated onboarding information."),
            )
        )


class FakeRouter:
    def route(self, question):
        del question
        return RoutingDecision(
            language=Language.HAUSA,
            tool=ToolName.KNOWLEDGE,
            retrieval_query="attendance requirement",
        )


class FakeAnswerer:
    def __init__(self):
        self.calls = 0

    def answer(self, *, question, language, evidence):
        self.calls += 1
        assert question
        assert language is Language.HAUSA
        assert "75%" in evidence
        return "Ana bukatar attendance na 75%."


class FakeVerifier:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0
        self.questions = []
        self.evidence = []

    def is_supported(self, *, question, evidence):
        self.calls += 1
        self.questions.append(question)
        self.evidence.append(evidence)
        assert question
        assert evidence
        if not self.verdicts:
            raise AssertionError("FakeVerifier has no verdict left.")
        return self.verdicts.pop(0)


def _settings(tmp_path):
    return Settings(
        data_dir=tmp_path,
        natlas_model="NCAIR1/N-ATLaS",
        natlas_device="auto",
        natlas_max_new_tokens=320,
        hf_token=None,
        top_k=3,
        min_retrieval_score=0.30,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.2:3b",
        default_version="v2",
    )


def test_v2_service_accepts_support_from_later_retrieved_passage(tmp_path):
    answerer = FakeAnswerer()
    verifier = FakeVerifier([False, True])
    service = AssistantService(
        "v2",
        settings=_settings(tmp_path),
        knowledge_base=FakeKnowledgeBase(),
        router=FakeRouter(),
        answerer=answerer,
        verifier=verifier,
    )

    response = service.chat("Attendance nawa nake bukata?")

    assert response.language is Language.HAUSA
    assert response.tool is ToolName.KNOWLEDGE
    assert response.sources == ("first.txt", "guide.txt", "other.txt")
    assert "75%" in response.answer
    assert verifier.calls == 2
    assert verifier.questions == ["attendance requirement", "attendance requirement"]
    assert verifier.evidence == [
        "[first.txt]\nRelated but insufficient information.",
        "[guide.txt]\nAttendance requires 75%.",
    ]
    assert answerer.calls == 1


def test_v2_service_stops_verifying_after_first_supported_passage(tmp_path):
    answerer = FakeAnswerer()
    verifier = FakeVerifier([True])
    service = AssistantService(
        "v2",
        settings=_settings(tmp_path),
        knowledge_base=FakeKnowledgeBase(),
        router=FakeRouter(),
        answerer=answerer,
        verifier=verifier,
    )

    service.chat("Attendance nawa nake bukata?")

    assert verifier.calls == 1
    assert verifier.evidence == ["[first.txt]\nRelated but insufficient information."]
    assert answerer.calls == 1


def test_v2_service_rejects_when_no_retrieved_passage_answers_question(tmp_path):
    answerer = FakeAnswerer()
    verifier = FakeVerifier([False, False, False])
    service = AssistantService(
        "v2",
        settings=_settings(tmp_path),
        knowledge_base=FakeKnowledgeBase(),
        router=FakeRouter(),
        answerer=answerer,
        verifier=verifier,
    )

    response = service.chat("Wane bayani ne jagorar ta tabbatar?")

    assert response.language is Language.HAUSA
    assert response.tool is ToolName.KNOWLEDGE
    assert response.sources == ()
    assert "Ban sami isasshen bayani" in response.answer
    assert verifier.calls == 3
    assert verifier.questions == [
        "attendance requirement",
        "attendance requirement",
        "attendance requirement",
    ]
    assert answerer.calls == 0
