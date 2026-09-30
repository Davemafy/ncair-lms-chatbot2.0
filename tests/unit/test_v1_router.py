from ncair_lms.models import PortalAction, ToolName
from ncair_lms.v1_router import V1KeywordRouter


def test_v1_preserves_keyword_portal_routing():
    decision = V1KeywordRouter().route("Where to sign in?")
    assert decision.tool is ToolName.PORTAL_LINK
    assert decision.action is PortalAction.LOGIN


def test_v1_falls_back_to_knowledge_for_policy_questions():
    decision = V1KeywordRouter().route("What attendance do I need?")
    assert decision.tool is ToolName.KNOWLEDGE
    assert decision.retrieval_query == "What attendance do I need?"
