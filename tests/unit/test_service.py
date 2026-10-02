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
    def __init__(self, *, supported=True):
        self.supported = supported
        self.queries = []

    def search(self, query, *, top_k):
        self.queries.append((query, top_k))
        return RetrievedEvidence(
            (
                EvidencePassage(
                    source="ncair_knowledge_base.txt",
                    text="Attendance Threshold: A minimum of 75% attendance is required.",
                ),
            ),
            support_verified=self.supported,
        )


class FakeRouter:
    def route(self, question):
        return RoutingDecision(
            language=Language.HAUSA,
            tool=ToolName.KNOWLEDGE,
            retrieval_query=question,
        )


class FakeAnswerer:
    def __init__(self):
        self.calls = []

    def answer(self, *, question, language, evidence):
        self.calls.append((question, language, evidence))
        return "Ana bukatar attendance na 75%."


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


def test_v2_service_trusts_calibrated_retrieval_support(tmp_path):
    answerer = FakeAnswerer()
    knowledge_base = FakeKnowledgeBase(supported=True)
    service = AssistantService(
        "v2",
        settings=_settings(tmp_path),
        knowledge_base=knowledge_base,
        router=FakeRouter(),
        answerer=answerer,
    )

    response = service.chat("Attendance nawa nake bukata?")

    assert response.language is Language.HAUSA
    assert response.tool is ToolName.KNOWLEDGE
    assert response.sources == ("ncair_knowledge_base.txt",)
    assert "75%" in response.answer
    assert knowledge_base.queries == [("Attendance nawa nake bukata?", 3)]
    assert len(answerer.calls) == 1


def test_v2_service_abstains_when_reranker_support_is_below_threshold(tmp_path):
    answerer = FakeAnswerer()
    service = AssistantService(
        "v2",
        settings=_settings(tmp_path),
        knowledge_base=FakeKnowledgeBase(supported=False),
        router=FakeRouter(),
        answerer=answerer,
    )

    response = service.chat("Nawa ake biyan allowance?")

    assert response.language is Language.HAUSA
    assert response.tool is ToolName.KNOWLEDGE
    assert response.sources == ()
    assert "Ban sami isasshen bayani" in response.answer
    assert answerer.calls == []
