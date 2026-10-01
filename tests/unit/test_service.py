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
            (EvidencePassage(source="guide.txt", text="Attendance requires 75%."),)
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
    def __init__(self, supported=True):
        self.supported = supported
        self.calls = 0

    def is_supported(self, *, question, evidence):
        self.calls += 1
        assert question
        assert evidence
        return self.supported


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


def test_v2_service_keeps_language_and_grounding(tmp_path):
    answerer = FakeAnswerer()
    verifier = FakeVerifier(supported=True)
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
    assert response.sources == ("guide.txt",)
    assert "75%" in response.answer
    assert verifier.calls == 1
    assert answerer.calls == 1


def test_v2_service_rejects_retrieval_that_does_not_answer_question(tmp_path):
    answerer = FakeAnswerer()
    verifier = FakeVerifier(supported=False)
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
    assert verifier.calls == 1
    assert answerer.calls == 0
