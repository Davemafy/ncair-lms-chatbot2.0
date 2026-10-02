from __future__ import annotations

import logging
from typing import Protocol

import requests

from .config import Settings
from .models import AssistantResponse, Language, RoutingDecision, ToolName, ToolResult
from .natlas import LocalNatlasClient, NatlasGroundedAnswerer, NatlasRouter
from .rag import AtomicKnowledgeBase, KnowledgeBase
from .tools import execute_tool
from .v1_router import V1KeywordRouter

LOGGER = logging.getLogger(__name__)

UNSUPPORTED_MESSAGES = {
    Language.ENGLISH: (
        "I could not verify that from the official NCAIR LMS guide. "
        "Please confirm it with an NCAIR facilitator."
    ),
    Language.HAUSA: (
        "Ban sami isasshen bayani a cikin jagorar NCAIR LMS na hukuma don tabbatar da "
        "wannan ba. Da fatan za ka tabbatar da shi da mai koyarwa na NCAIR."
    ),
    Language.YORUBA: (
        "Mi ò rí ìsọfúnni tó tó nínú ìtọ́sọ́nà NCAIR LMS àṣẹ láti jẹ́rìí èyí. "
        "Jọ̀wọ́ jẹ́rìí rẹ̀ pẹ̀lú olùkọ́ NCAIR."
    ),
    Language.IGBO: (
        "Enweghị m ozi zuru ezu n’ime ntuziaka NCAIR LMS gọọmenti iji kwado nke a. "
        "Biko gosi ya n’aka onye nkuzi NCAIR."
    ),
}


class Router(Protocol):
    def route(self, question: str) -> RoutingDecision: ...


class GroundedAnswerer(Protocol):
    def answer(self, *, question: str, language: Language, evidence: str) -> str: ...


class KnowledgeSearcher(Protocol):
    def search(self, query: str, *, top_k: int): ...


class OllamaAnswerer:
    def __init__(self, settings: Settings):
        self._settings = settings

    def answer(self, *, question: str, language: Language, evidence: str) -> str:
        del language
        prompt = (
            "You are the official NCAIR LMS Onboarding Assistant. "
            "Answer the user's question concisely using only the official context below. "
            "If the context is insufficient, say you could not verify the answer.\n\n"
            f"OFFICIAL CONTEXT:\n{evidence}\n\n"
            f"QUESTION:\n{question}\n\nANSWER:"
        )
        response = requests.post(
            f"{self._settings.ollama_base_url}/api/generate",
            json={
                "model": self._settings.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        if not answer:
            raise ValueError("Ollama returned an empty answer.")
        return answer


class AssistantService:
    def __init__(
        self,
        version: str,
        *,
        settings: Settings | None = None,
        knowledge_base: KnowledgeSearcher | None = None,
        router: Router | None = None,
        answerer: GroundedAnswerer | None = None,
    ):
        if version not in {"v1", "v2"}:
            raise ValueError("version must be 'v1' or 'v2'.")

        self.version = version
        self.settings = settings or Settings.from_env()

        if knowledge_base is not None:
            self.knowledge_base = knowledge_base
        elif version == "v1":
            self.knowledge_base = KnowledgeBase(
                self.settings.data_dir,
                min_score=self.settings.min_retrieval_score,
                embedding_device=self.settings.embedding_device,
            )
        else:
            self.knowledge_base = AtomicKnowledgeBase(
                self.settings.data_dir,
                embedding_model=self.settings.v2_embedding_model,
                embedding_device=self.settings.embedding_device,
                reranker_model=self.settings.reranker_model,
                reranker_device=self.settings.reranker_device,
                candidate_k=self.settings.rerank_candidates,
                min_score=self.settings.rerank_min_score,
                min_margin=self.settings.rerank_min_margin,
            )

        natlas_client = None
        if version == "v2" and (router is None or answerer is None):
            natlas_client = LocalNatlasClient(self.settings)

        if router is not None:
            self.router = router
        elif version == "v1":
            self.router = V1KeywordRouter()
        else:
            assert natlas_client is not None
            self.router = NatlasRouter(natlas_client)

        if answerer is not None:
            self.answerer = answerer
        elif version == "v1":
            self.answerer = OllamaAnswerer(self.settings)
        else:
            assert natlas_client is not None
            self.answerer = NatlasGroundedAnswerer(natlas_client, self.settings)

    def route(self, question: str) -> RoutingDecision:
        decision = self.router.route(question)
        LOGGER.info(
            "router_selection version=%s tool=%s language=%s",
            self.version,
            decision.tool.value,
            decision.language.value,
        )
        return decision

    def execute(self, decision: RoutingDecision, *, question: str | None = None) -> ToolResult:
        del question
        return execute_tool(
            decision,
            self.knowledge_base,
            top_k=self.settings.top_k,
            allow_full_step_sequence=self.version == "v1",
        )

    def respond(
        self,
        *,
        question: str,
        decision: RoutingDecision,
        tool_result: ToolResult | None = None,
    ) -> AssistantResponse:
        result = tool_result or self.execute(decision)

        if decision.tool is ToolName.KNOWLEDGE and not result.evidence.supported:
            answer = UNSUPPORTED_MESSAGES[decision.language]
        elif self.version == "v1" and decision.tool is not ToolName.KNOWLEDGE:
            answer = result.answer
        else:
            evidence = result.evidence.as_context()
            answer = self.answerer.answer(
                question=question,
                language=decision.language,
                evidence=evidence,
            )

        sources = result.evidence.sources
        if decision.tool is ToolName.KNOWLEDGE and not result.evidence.supported:
            sources = ()

        return AssistantResponse(
            answer=answer,
            language=decision.language,
            tool=decision.tool,
            sources=sources,
        )

    def chat(self, question: str) -> AssistantResponse:
        decision = self.route(question)
        tool_result = self.execute(decision, question=question)
        return self.respond(question=question, decision=decision, tool_result=tool_result)
