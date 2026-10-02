from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from .config import Settings
from .errors import InvalidModelOutputError, InvalidToolArgumentsError, ModelUnavailableError
from .models import Language, PortalAction, RoutingDecision, RoutingStatus, ToolName

LOGGER = logging.getLogger(__name__)

LANGUAGE_SYSTEM_PROMPT = """Classify the user's input language for the NCAIR LMS assistant.
Return exactly one JSON object and no prose:

{"language":"english|hausa|yoruba|igbo"}

Rules:
- Choose exactly one of english, hausa, yoruba, or igbo.
- Identify the language from the sentence's grammar and function words.
- Do not let names, URLs, acronyms, course names, or English technical terms determine the label.
- For code-switched text, choose the language carrying most of the sentence grammar.
- Do not answer, translate, or route the request.
"""

TOOL_ROUTER_SYSTEM_PROMPT = """You are the semantic tool router for the NCAIR LMS assistant.
The user's language is supplied separately. Do not classify language.
Return exactly one JSON object and no prose.

Allowed tools:
1. get_portal_link
   arguments: {"action":"<one allowed action>"}
   allowed actions: main, login, signin, ncair_home, register, profile, courses,
   track_selection, support
2. get_step_guidance
   arguments: {"step":1|2|3|4}
3. search_ncair_knowledge_base
   arguments: {"query":"<concise English retrieval query>"}

Routing rules:
- Choose exactly one tool.
- Use get_portal_link only when the user is asking to open, find, visit, or navigate to a
  specific NCAIR/LMS page or destination.
- Portal actions must be one of the exact allowed action values above.
- Use get_step_guidance only when the user explicitly refers to numbered onboarding step
  1, 2, 3, or 4.
- Use search_ncair_knowledge_base for factual questions, policies, requirements,
  troubleshooting, general onboarding, claims to verify, and facts that may be undocumented.
- For search_ncair_knowledge_base, write the retrieval query in concise English even when
  the user wrote in another supported language.
- Never invent a portal action, tool, or step number.
"""

EVIDENCE_SYSTEM_PROMPT = """Decide whether one official NCAIR passage directly answers the
retrieval question. Return exactly one JSON object and no prose:

{"supported":true|false,"quote":"<exact excerpt from the passage or empty string>"}

Rules:
- Use only the supplied passage. Do not use outside knowledge.
- Return true only when the passage explicitly states the requested fact or directly corrects
  the claim in the question.
- When supported is true, quote must copy an exact excerpt from the supplied passage that
  contains the supporting fact. Do not paraphrase the quote.
- When supported is false, quote must be an empty string.
- Different wording in the question is fine; the requested fact itself must be present.
- Return false for mere topical similarity, missing facts, or ambiguous evidence.
"""

TOOL_SUFFICIENCY_SYSTEM_PROMPT = """Check whether the selected non-knowledge tool directly
satisfies the user's requested outcome. Return exactly one JSON object and no prose:

{"sufficient":true|false}

Rules:
- The candidate tool has already been selected by the semantic router.
- Return true when the user asked for that destination/page or explicitly asked about that
  numbered onboarding step.
- Return false only when executing the candidate would fail to provide the information the
  user actually requested, such as a policy, requirement, explanation, troubleshooting fact,
  or claim verification.
- Judge the requested outcome, not merely the topic words.
- Do not choose another tool and do not answer the request.
"""

KNOWLEDGE_CHALLENGE_SYSTEM_PROMPT = """Act as a conservative escalation guard for the NCAIR
LMS assistant. A direct router selected a portal or numbered-step tool and a first verifier
said that tool may be insufficient. Decide whether a knowledge-base answer is actually
required. Return exactly one JSON object and no prose:

{"knowledge_required":true|false}

Rules:
- Return false when the user is asking for a page, URL, destination, or the selected numbered
  onboarding step itself.
- Return true when the user needs a factual answer, policy, requirement, explanation,
  troubleshooting fact, or claim verification rather than merely the destination/step.
- If the request can be satisfied by executing the candidate tool as stated, return false.
- Do not answer the request and do not choose a different portal action or step.
"""

KNOWLEDGE_QUERY_SYSTEM_PROMPT = """Convert the user's information need into one concise
English retrieval query for the official NCAIR knowledge base. Return exactly one JSON object
and no prose:

{"query":"<concise English retrieval query>"}

Rules:
- Preserve the user's actual information need, including numbers, constraints, actor type,
  requested attribute, and named NCAIR/LMS entities.
- Translate the information need into concise English when necessary.
- Prefer a short factual search phrase over a conversational sentence.
- Do not answer the question.
- Do not return a portal action or onboarding step.
"""


class NatlasTextClient(Protocol):
    def generate(self, messages: Sequence[dict[str, str]], *, max_new_tokens: int) -> str: ...


class LocalNatlasClient:
    """Lazy local adapter for the official NCAIR1/N-ATLaS Hugging Face model."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ModelUnavailableError(
                "N-ATLaS requires transformers, accelerate, and torch. "
                "Install the runtime dependencies first."
            ) from exc

        kwargs = {"device_map": self._settings.natlas_device, "torch_dtype": "auto"}
        if self._settings.natlas_quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["torch_dtype"] = torch.float16
        if self._settings.hf_token:
            kwargs["token"] = self._settings.hf_token

        LOGGER.info(
            "loading_natlas model=%s quantization=%s",
            self._settings.natlas_model,
            self._settings.natlas_quantization,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._settings.natlas_model,
                token=self._settings.hf_token,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self._settings.natlas_model,
                **kwargs,
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load N-ATLaS model {self._settings.natlas_model!r}."
            ) from exc

        self._tokenizer = tokenizer
        self._model = model

    def generate(self, messages: Sequence[dict[str, str]], *, max_new_tokens: int) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        try:
            prompt = self._tokenizer.apply_chat_template(
                list(messages),
                add_generation_prompt=True,
                tokenize=False,
                date_string=datetime.now().strftime("%d %b %Y"),
            )
            inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            model_device = next(self._model.parameters()).device
            inputs = {name: value.to(model_device) for name, value in inputs.items()}
            prompt_length = inputs["input_ids"].shape[-1]

            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.05,
            )
            generated = output[0][prompt_length:]
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exc:
            raise ModelUnavailableError("N-ATLaS inference failed.") from exc


def _close_truncated_json(text: str) -> str | None:
    """Repair only a JSON object that ends with missing closing braces/brackets."""
    stack: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (opener, char) not in {("{", "}"), ("[", "]")}:
                return None

    if in_string or not stack or len(stack) > 2:
        return None

    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[opener] for opener in reversed(stack))


def _json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    start = text.find("{")
    if start < 0:
        raise InvalidModelOutputError("N-ATLaS did not return a JSON object.")

    candidate = text[start:]
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as exc:
        repaired = _close_truncated_json(candidate)
        if repaired is None:
            raise InvalidModelOutputError("N-ATLaS returned invalid JSON.") from exc
        try:
            value, _ = json.JSONDecoder().raw_decode(repaired)
        except json.JSONDecodeError as repaired_exc:
            raise InvalidModelOutputError("N-ATLaS returned invalid JSON.") from repaired_exc
        LOGGER.warning("natlas_repaired_truncated_json raw=%r", candidate[:1000])

    if not isinstance(value, dict):
        raise InvalidModelOutputError("N-ATLaS output must be a JSON object.")
    return value





_ORTHOGRAPHIC_LANGUAGE_MARKERS = {
    Language.HAUSA: frozenset({"ƙ", "ɗ", "ɓ", "ƴ", "ƙ".upper(), "ɗ".upper(), "ɓ".upper()}),
    Language.YORUBA: frozenset({"ṣ", "Ṣ", "ẹ", "Ẹ"}),
    Language.IGBO: frozenset({"ị", "Ị", "ụ", "Ụ"}),
}


def _orthographic_language_hint(question: str) -> Language | None:
    """Return a language only when orthography gives one unambiguous supported-language hint."""
    matches = {
        language
        for language, markers in _ORTHOGRAPHIC_LANGUAGE_MARKERS.items()
        if any(marker in question for marker in markers)
    }
    if len(matches) == 1:
        return next(iter(matches))
    return None


class NatlasLanguageDetector:
    def __init__(self, client: NatlasTextClient):
        self._client = client

    @staticmethod
    def _parse(raw: str) -> Language:
        try:
            payload = _json_object(raw)
            return Language(str(payload["language"]).lower())
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidModelOutputError("N-ATLaS returned an invalid language label.") from exc

    @staticmethod
    def _reconcile(question: str, detected: Language) -> Language:
        hint = _orthographic_language_hint(question)
        if hint is not None and hint is not detected:
            LOGGER.info(
                "natlas_language_orthography_override detected=%s override=%s",
                detected.value,
                hint.value,
            )
            return hint
        return detected

    def detect(self, question: str) -> Language:
        messages = [
            {"role": "system", "content": LANGUAGE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        raw = self._client.generate(messages, max_new_tokens=40)

        try:
            return self._reconcile(question, self._parse(raw))
        except InvalidModelOutputError as first_error:
            LOGGER.warning("natlas_invalid_language_output raw=%r", raw[:1000])
            retry = self._client.generate(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "The previous output was invalid. Return only the required JSON with "
                            "one allowed language value: english, hausa, yoruba, or igbo."
                        ),
                    },
                ],
                max_new_tokens=40,
            )
            try:
                return self._reconcile(question, self._parse(retry))
            except InvalidModelOutputError as retry_error:
                LOGGER.warning("natlas_invalid_language_retry raw=%r", retry[:1000])
                raise retry_error from first_error


def _normalized_tool_call(raw: str) -> tuple[ToolName, dict]:
    """Normalize equivalent structured tool-call serializations emitted by N-ATLaS."""
    payload = _json_object(raw)
    tool_values = {tool.value for tool in ToolName}
    portal_values = {action.value for action in PortalAction}

    try:
        if "tool" in payload:
            tool = ToolName(str(payload["tool"]))
            arguments = payload.get("arguments")
            if arguments is None:
                arguments = {
                    key: value for key, value in payload.items() if key not in {"tool", "language"}
                }
        elif "action" in payload and str(payload["action"]) in tool_values:
            tool = ToolName(str(payload["action"]))
            arguments = {
                key: value for key, value in payload.items() if key not in {"action", "language"}
            }
        elif "action" in payload and str(payload["action"]).lower() in portal_values:
            tool = ToolName.PORTAL_LINK
            arguments = {"action": str(payload["action"]).lower()}
        else:
            keyed_tools = [key for key in payload if key in tool_values]
            if len(keyed_tools) != 1:
                raise InvalidModelOutputError(
                    "N-ATLaS routing output must identify one documented tool."
                )
            tool = ToolName(keyed_tools[0])
            arguments = payload[keyed_tools[0]]
    except (TypeError, ValueError) as exc:
        raise InvalidModelOutputError("N-ATLaS routing output has an invalid tool.") from exc

    if not isinstance(arguments, dict):
        raise InvalidModelOutputError("N-ATLaS tool arguments must be an object.")

    return tool, arguments


def _parse_sufficiency(raw: str) -> bool:
    payload = _json_object(raw)
    sufficient = payload.get("sufficient")
    if not isinstance(sufficient, bool):
        raise InvalidModelOutputError("N-ATLaS tool-sufficiency verdict must contain a boolean.")
    return sufficient


def _parse_knowledge_query(raw: str) -> str:
    payload = _json_object(raw)
    query = payload.get("query")

    if query is None:
        tool_values = {tool.value for tool in ToolName}
        if payload.get("tool") == ToolName.KNOWLEDGE.value:
            arguments = payload.get("arguments")
            if isinstance(arguments, dict):
                query = arguments.get("query")
        elif payload.get("action") == ToolName.KNOWLEDGE.value:
            query = payload.get("query")
        elif ToolName.KNOWLEDGE.value in payload:
            arguments = payload[ToolName.KNOWLEDGE.value]
            if isinstance(arguments, dict):
                query = arguments.get("query")
        elif any(key in payload for key in tool_values):
            raise InvalidModelOutputError(
                "N-ATLaS knowledge fallback returned the wrong tool family."
            )

    if not isinstance(query, str) or not query.strip():
        raise InvalidModelOutputError("N-ATLaS knowledge query must be a non-empty string.")
    return query.strip()


def _parse_tool_decision(raw: str, *, language: Language) -> RoutingDecision:
    tool, arguments = _normalized_tool_call(raw)

    try:
        if tool is ToolName.PORTAL_LINK:
            action = PortalAction(str(arguments["action"]).lower())
            decision = RoutingDecision(
                language=language,
                tool=tool,
                action=action,
                status=RoutingStatus.MODEL,
            )
        elif tool is ToolName.STEP_GUIDANCE:
            step = arguments.get("step")
            if isinstance(step, bool) or not isinstance(step, int):
                raise InvalidModelOutputError("get_step_guidance.step must be an integer.")
            decision = RoutingDecision(
                language=language,
                tool=tool,
                step=step,
                status=RoutingStatus.MODEL,
            )
        else:
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise InvalidModelOutputError(
                    "search_ncair_knowledge_base.query must be a non-empty string."
                )
            decision = RoutingDecision(
                language=language,
                tool=tool,
                retrieval_query=query.strip(),
                status=RoutingStatus.MODEL,
            )

        return decision.validate()
    except (KeyError, ValueError, InvalidToolArgumentsError) as exc:
        raise InvalidModelOutputError(str(exc)) from exc


class NatlasRouter:
    def __init__(
        self,
        client: NatlasTextClient,
        language_detector: NatlasLanguageDetector | None = None,
    ):
        self._client = client
        self._language_detector = language_detector or NatlasLanguageDetector(client)

    def _route_once(self, question: str, *, language: Language) -> RoutingDecision:
        messages = [
            {"role": "system", "content": TOOL_ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Input language: {language.value}\nUser request:\n{question}",
            },
        ]
        raw = self._client.generate(messages, max_new_tokens=180)

        try:
            return _parse_tool_decision(raw, language=language)
        except InvalidModelOutputError as first_error:
            LOGGER.warning("natlas_invalid_route_output raw=%r", raw[:1000])
            retry = self._client.generate(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "The previous tool decision was invalid. Correct it using only the "
                            "documented tool schema and exact allowed enum values. "
                            f"Validation error: {first_error}"
                        ),
                    },
                ],
                max_new_tokens=180,
            )
            try:
                return _parse_tool_decision(retry, language=language)
            except InvalidModelOutputError as retry_error:
                LOGGER.warning("natlas_invalid_route_retry raw=%r", retry[:1000])
                raise retry_error from first_error

    def _tool_is_sufficient(self, question: str, decision: RoutingDecision) -> bool:
        if decision.tool is ToolName.KNOWLEDGE:
            return True

        if decision.tool is ToolName.PORTAL_LINK:
            candidate = f"get_portal_link(action={decision.action.value})"
        else:
            candidate = f"get_step_guidance(step={decision.step})"

        messages = [
            {"role": "system", "content": TOOL_SUFFICIENCY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (f"User request:\n{question}\n\nCandidate tool call:\n{candidate}"),
            },
        ]
        raw = self._client.generate(messages, max_new_tokens=40)

        try:
            return _parse_sufficiency(raw)
        except InvalidModelOutputError as first_error:
            LOGGER.warning("natlas_invalid_tool_sufficiency raw=%r", raw[:1000])
            retry = self._client.generate(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "The previous verdict was invalid. Return only "
                            '{"sufficient":true} or {"sufficient":false}.'
                        ),
                    },
                ],
                max_new_tokens=40,
            )
            try:
                return _parse_sufficiency(retry)
            except InvalidModelOutputError as retry_error:
                LOGGER.warning("natlas_invalid_tool_sufficiency_retry raw=%r", retry[:1000])
                raise retry_error from first_error

    def _knowledge_fallback(
        self,
        question: str,
        *,
        language: Language,
    ) -> RoutingDecision:
        messages = [
            {"role": "system", "content": KNOWLEDGE_QUERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Input language: {language.value}\nUser request:\n{question}",
            },
        ]
        raw = self._client.generate(messages, max_new_tokens=100)

        try:
            query = _parse_knowledge_query(raw)
        except InvalidModelOutputError as first_error:
            LOGGER.warning("natlas_invalid_knowledge_query raw=%r", raw[:1000])
            retry = self._client.generate(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "The previous retrieval query was invalid. Return only a JSON object "
                            'with one non-empty English "query" string.'
                        ),
                    },
                ],
                max_new_tokens=100,
            )
            try:
                query = _parse_knowledge_query(retry)
            except InvalidModelOutputError as retry_error:
                LOGGER.warning("natlas_invalid_knowledge_query_retry raw=%r", retry[:1000])
                raise retry_error from first_error

        return RoutingDecision(
            language=language,
            tool=ToolName.KNOWLEDGE,
            retrieval_query=query,
            status=RoutingStatus.MODEL,
        ).validate()

    def route(self, question: str) -> RoutingDecision:
        language = self._language_detector.detect(question)
        decision = self._route_once(question, language=language)

        if self._tool_is_sufficient(question, decision):
            return decision

        LOGGER.info(
            "natlas_tool_escalation selected_tool=%s language=%s",
            decision.tool.value,
            language.value,
        )
        return self._knowledge_fallback(question, language=language)


class NatlasEvidenceVerifier:
    def __init__(self, client: NatlasTextClient):
        self._client = client

    @staticmethod
    def _parse(raw: str) -> bool:
        payload = _json_object(raw)
        supported = payload.get("supported")
        if not isinstance(supported, bool):
            raise InvalidModelOutputError("N-ATLaS evidence verdict must contain a boolean.")
        return supported

    def is_supported(self, *, question: str, evidence: str) -> bool:
        messages = [
            {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nOfficial evidence:\n{evidence}",
            },
        ]
        raw = self._client.generate(messages, max_new_tokens=40)

        try:
            return self._parse(raw)
        except InvalidModelOutputError as first_error:
            LOGGER.warning("natlas_invalid_evidence_verdict raw=%r", raw[:1000])
            retry = self._client.generate(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "The previous verdict was invalid. Return only "
                            '{"supported":true} or {"supported":false}.'
                        ),
                    },
                ],
                max_new_tokens=40,
            )
            try:
                return self._parse(retry)
            except InvalidModelOutputError as retry_error:
                LOGGER.warning("natlas_invalid_evidence_retry raw=%r", retry[:1000])
                raise retry_error from first_error


class NatlasGroundedAnswerer:
    def __init__(self, client: NatlasTextClient, settings: Settings):
        self._client = client
        self._settings = settings

    def answer(self, *, question: str, language: Language, evidence: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the NCAIR LMS Assistant. Answer only from the supplied official "
                    "evidence. Do not invent dates, contacts, policies, URLs, or requirements. "
                    f"Answer in {language.value}. Preserve every URL exactly as written. "
                    "Be concise. If the evidence is insufficient, say you could not verify the "
                    "answer from the official guide."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nOfficial evidence:\n{evidence}",
            },
        ]
        answer = self._client.generate(
            messages,
            max_new_tokens=self._settings.natlas_max_new_tokens,
        ).strip()
        if not answer:
            raise InvalidModelOutputError("N-ATLaS returned an empty answer.")
        return answer
