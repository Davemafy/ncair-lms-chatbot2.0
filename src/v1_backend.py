"""Compatibility layer for code that imported the original V1 module directly."""

from ncair_lms.config import Settings
from ncair_lms.rag import KnowledgeBase
from ncair_lms.service import AssistantService
from ncair_lms.tools import get_portal_link as _get_portal_link
from ncair_lms.tools import get_step_guidance as _get_step_guidance

SETTINGS = Settings.from_env()
OLLAMA_BASE_URL = SETTINGS.ollama_base_url
OLLAMA_MODEL = SETTINGS.ollama_model
_KNOWLEDGE_BASE = KnowledgeBase(
    SETTINGS.data_dir,
    min_score=SETTINGS.min_retrieval_score,
)


def search_ncair_knowledge_base(query: str, top_k: int = 2) -> str:
    return _KNOWLEDGE_BASE.search(query, top_k=top_k).as_context()


def get_portal_link(action: str) -> str:
    return _get_portal_link(action).answer


def get_step_guidance(step_number: int) -> str:
    return _get_step_guidance(step_number).answer


def agentic_rag_orchestrator(user_query: str) -> dict:
    return AssistantService(
        "v1",
        settings=SETTINGS,
        knowledge_base=_KNOWLEDGE_BASE,
    ).chat(user_query).as_dict()
