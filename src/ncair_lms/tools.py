from __future__ import annotations

from .errors import InvalidToolArgumentsError
from .models import (
    EvidencePassage,
    PortalAction,
    RetrievedEvidence,
    RoutingDecision,
    ToolName,
    ToolResult,
)
from .rag import KnowledgeBase

PORTAL_URLS: dict[PortalAction, str] = {
    PortalAction.MAIN: "https://lms.ncair.nitda.gov.ng",
    PortalAction.LOGIN: "https://lms.ncair.nitda.gov.ng/intern/signin",
    PortalAction.SIGNIN: "https://lms.ncair.nitda.gov.ng/intern/signin",
    PortalAction.NCAIR_HOME: "https://ncair.nitda.gov.ng/",
    PortalAction.REGISTER: "https://lms.ncair.nitda.gov.ng",
    PortalAction.PROFILE: "https://lms.ncair.nitda.gov.ng/intern/profile",
    PortalAction.COURSES: "https://lms.ncair.nitda.gov.ng/intern/courses",
    PortalAction.TRACK_SELECTION: "https://lms.ncair.nitda.gov.ng/intern/courses",
    PortalAction.SUPPORT: "https://ncair.nitda.gov.ng/contact/",
}

PORTAL_LABELS: dict[PortalAction, str] = {
    PortalAction.MAIN: "NCAIR LMS",
    PortalAction.LOGIN: "NCAIR LMS sign-in page",
    PortalAction.SIGNIN: "NCAIR LMS sign-in page",
    PortalAction.NCAIR_HOME: "NCAIR website",
    PortalAction.REGISTER: "NCAIR LMS registration page",
    PortalAction.PROFILE: "intern profile page",
    PortalAction.COURSES: "courses page",
    PortalAction.TRACK_SELECTION: "track selection page",
    PortalAction.SUPPORT: "NCAIR support page",
}

STEP_GUIDANCE: dict[int, str] = {
    1: (
        "Step 1 — Attend orientation: new interns begin with the physical orientation at NCAIR."
    ),
    2: (
        "Step 2 — Register physically: complete registration with facilitators in the "
        "PSIN 50-Seater Hall."
    ),
    3: (
        "Step 3 — Complete your LMS account: use the invitation email, create a valid "
        "password, and upload the requested document in PDF or PNG under 2 MB."
    ),
    4: (
        "Step 4 — Begin your assigned courses: confirm your cohort and course selection "
        "in the LMS. Registration closes at 5:00 PM on registration day."
    ),
}


def _portal_action(value: PortalAction | str) -> PortalAction:
    if isinstance(value, PortalAction):
        return value
    try:
        return PortalAction(value.strip().lower())
    except ValueError as exc:
        raise InvalidToolArgumentsError(f"Unknown portal action: {value!r}") from exc


def get_portal_link(action: PortalAction | str) -> ToolResult:
    action = _portal_action(action)
    url = PORTAL_URLS[action]
    answer = f"[Open the {PORTAL_LABELS[action]}]({url})."
    evidence = RetrievedEvidence(
        (EvidencePassage(source="Verified NCAIR portal registry", text=answer),)
    )
    return ToolResult(answer=answer, evidence=evidence)


def get_step_guidance(step: int | None) -> ToolResult:
    if step is None:
        answer = "\n\n".join(STEP_GUIDANCE[number] for number in sorted(STEP_GUIDANCE))
    else:
        if step not in STEP_GUIDANCE:
            raise InvalidToolArgumentsError("Step must be one of 1, 2, 3, or 4.")
        answer = STEP_GUIDANCE[step]

    evidence = RetrievedEvidence(
        (EvidencePassage(source="Official NCAIR onboarding sequence", text=answer),)
    )
    return ToolResult(answer=answer, evidence=evidence)


def execute_tool(
    decision: RoutingDecision,
    knowledge_base: KnowledgeBase,
    *,
    top_k: int,
    allow_full_step_sequence: bool = False,
) -> ToolResult:
    decision.validate(allow_full_step_sequence=allow_full_step_sequence)

    if decision.tool is ToolName.PORTAL_LINK:
        assert decision.action is not None
        return get_portal_link(decision.action)

    if decision.tool is ToolName.STEP_GUIDANCE:
        return get_step_guidance(decision.step)

    if decision.tool is ToolName.KNOWLEDGE:
        assert decision.retrieval_query is not None
        evidence = knowledge_base.search(decision.retrieval_query, top_k=top_k)
        return ToolResult(answer="", evidence=evidence)

    raise InvalidToolArgumentsError(f"Unsupported tool: {decision.tool.value}")
