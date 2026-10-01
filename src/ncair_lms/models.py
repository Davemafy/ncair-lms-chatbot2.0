from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .errors import InvalidToolArgumentsError


class Language(StrEnum):
    ENGLISH = "english"
    HAUSA = "hausa"
    YORUBA = "yoruba"
    IGBO = "igbo"


class ToolName(StrEnum):
    PORTAL_LINK = "get_portal_link"
    STEP_GUIDANCE = "get_step_guidance"
    KNOWLEDGE = "search_ncair_knowledge_base"


class PortalAction(StrEnum):
    MAIN = "main"
    LOGIN = "login"
    SIGNIN = "signin"
    NCAIR_HOME = "ncair_home"
    REGISTER = "register"
    PROFILE = "profile"
    COURSES = "courses"
    TRACK_SELECTION = "track_selection"
    SUPPORT = "support"


class RoutingStatus(StrEnum):
    MODEL = "model"
    KEYWORD_BASELINE = "keyword_baseline"


@dataclass(frozen=True)
class RoutingDecision:
    language: Language
    tool: ToolName
    action: PortalAction | None = None
    step: int | None = None
    retrieval_query: str | None = None
    status: RoutingStatus = RoutingStatus.MODEL

    def validate(self, *, allow_full_step_sequence: bool = False) -> RoutingDecision:
        if self.tool is ToolName.PORTAL_LINK:
            if self.action is None:
                raise InvalidToolArgumentsError("get_portal_link requires an action.")
            return self

        if self.tool is ToolName.STEP_GUIDANCE:
            if self.step is None and allow_full_step_sequence:
                return self
            if self.step not in {1, 2, 3, 4}:
                raise InvalidToolArgumentsError("get_step_guidance requires step 1, 2, 3, or 4.")
            return self

        if self.tool is ToolName.KNOWLEDGE:
            if not self.retrieval_query or not self.retrieval_query.strip():
                raise InvalidToolArgumentsError(
                    "search_ncair_knowledge_base requires a non-empty retrieval query."
                )
            return self

        raise InvalidToolArgumentsError(f"Unsupported tool: {self.tool}")


@dataclass(frozen=True)
class EvidencePassage:
    source: str
    text: str
    page: int | None = None

    @property
    def citation(self) -> str:
        return f"{self.source}, page {self.page}" if self.page else self.source


@dataclass(frozen=True)
class RetrievedEvidence:
    passages: tuple[EvidencePassage, ...] = ()
    support_verified: bool | None = None

    @property
    def supported(self) -> bool:
        if self.support_verified is not None:
            return self.support_verified
        return bool(self.passages)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(passage.citation for passage in self.passages))

    def as_context(self) -> str:
        return "\n\n---\n\n".join(
            f"[{passage.citation}]\n{passage.text.strip()}" for passage in self.passages
        )


@dataclass(frozen=True)
class ToolResult:
    answer: str
    evidence: RetrievedEvidence


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    language: Language
    tool: ToolName
    sources: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "language": self.language.value,
            "tool": self.tool.value,
            "sources": list(self.sources),
        }
