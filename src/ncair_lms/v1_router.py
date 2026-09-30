from __future__ import annotations

from .models import Language, PortalAction, RoutingDecision, RoutingStatus, ToolName

PORTAL_HINTS = ("link", "url", "where to", "open", "page", "access", "portal")


class V1KeywordRouter:
    """Preserved keyword-routing baseline used for the V1 comparison."""

    def route(self, question: str) -> RoutingDecision:
        query = question.lower()

        if any(hint in query for hint in PORTAL_HINTS):
            if "register" in query or "sign up" in query:
                action = PortalAction.REGISTER
            elif "log" in query or "sign in" in query:
                action = PortalAction.LOGIN
            elif "profile" in query:
                action = PortalAction.PROFILE
            elif "course" in query:
                action = PortalAction.COURSES
            elif "track" in query:
                action = PortalAction.TRACK_SELECTION
            elif "support" in query or "help" in query:
                action = PortalAction.SUPPORT
            else:
                action = PortalAction.MAIN

            return RoutingDecision(
                language=Language.ENGLISH,
                tool=ToolName.PORTAL_LINK,
                action=action,
                status=RoutingStatus.KEYWORD_BASELINE,
            )

        if "stuck" in query or "step" in query:
            if "1" in query or "profile" in query:
                step = 1
            elif "2" in query or "upload" in query or "id" in query:
                step = 2
            elif "3" in query or "track" in query:
                step = 3
            elif "4" in query or "submit" in query:
                step = 4
            else:
                step = None

            return RoutingDecision(
                language=Language.ENGLISH,
                tool=ToolName.STEP_GUIDANCE,
                step=step,
                status=RoutingStatus.KEYWORD_BASELINE,
            )

        return RoutingDecision(
            language=Language.ENGLISH,
            tool=ToolName.KNOWLEDGE,
            retrieval_query=question,
            status=RoutingStatus.KEYWORD_BASELINE,
        )
