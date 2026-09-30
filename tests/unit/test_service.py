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
    def answer(self, *, question, language, evidence):
        assert question
        assert language is Language.HAUSA
        assert "75%" in evidence
        return "Ana bukatar attendance na 75%."


def test_v2_service_keeps_language_and_grounding(tmp_path):
    settings = Settings(
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
    service = AssistantService(
        "v2",
        settings=settings,
        knowledge_base=FakeKnowledgeBase(),
        router=FakeRouter(),
        answerer=FakeAnswerer(),
    )
    response = service.chat("Attendance nawa nake bukata?")
    assert response.language is Language.HAUSA
    assert response.tool is ToolName.KNOWLEDGE
    assert response.sources == ("guide.txt",)
    assert "75%" in response.answer
