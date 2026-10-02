from __future__ import annotations


import json
import logging
import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from .config import Settings
from .errors import InvalidModelOutputError, ModelUnavailableError
from .models import Language, PortalAction, RoutingDecision, RoutingStatus, ToolName

LOGGER = logging.getLogger(__name__)

LANGUAGE_SYSTEM_PROMPT = """Identify the language carrying the grammar of the user request.
Return exactly one JSON object and no prose:
{"language":"english|hausa|yoruba|igbo"}

Ignore names, URLs, acronyms, course names, and borrowed English technical words. For
code-switched text, choose the language carrying most of the sentence structure.
"""

TOOL_SYSTEM_PROMPT = """Choose which NCAIR LMS capability directly matches the user's requested
outcome. Return exactly one JSON object and no prose:
{"tool":"get_portal_link|get_step_guidance|search_ncair_knowledge_base"}

Tool contracts:
- get_portal_link: only when the user wants a page, URL, link, website, or portal destination.
- get_step_guidance: only when the user explicitly asks about numbered onboarding step 1-4.
- search_ncair_knowledge_base: facts, policies, requirements, schedules, explanations,
  troubleshooting, claim verification, and questions whose answer may be undocumented.

Classify the requested outcome rather than topic words. A question about login, registration,
courses, profile, support, or another portal topic is still knowledge search when the user wants
information about it rather than the page itself.
"""

PORTAL_SYSTEM_PROMPT = """The user explicitly wants an NCAIR/LMS destination. Return exactly one
JSON object and no prose:
{"action":"main|login|ncair_home|register|profile|courses|track_selection|support"}
Choose the destination that best matches the request. Do not answer the request.
"""

EVIDENCE_SYSTEM_PROMPT = """Decide whether the supplied official NCAIR evidence directly answers
the user's information need. Return exactly one JSON object and no prose:
{"supported":true|false}

Use only the supplied evidence. Return true when the requested fact is explicitly present or the
evidence directly corrects the user's claim. Different wording is fine. Return false for topical
similarity, missing facts, guesses, or evidence that does not actually answer the question.
"""

_LANGUAGE_LEXICON = {
    Language.ENGLISH: frozenset(
        "where what how when which who why do does did should can could would need open show take "
        "go visit page link website sign login home my the is are list official happens explain "
        "right from only use after before".split()  # noqa: SIM905
    ),
    Language.HAUSA: frozenset(
        "ina zan shiga bude buɗe min shafi shafin zaben zaɓen son me yi mataki ka bayyana kashi "
        "nawa nake bukata buƙata domin wuce wane irin idan hakan nufi ake bayarwa matsayin menene "
        "dole ko karfe akwai sabon sabo daga amma ba sai dina yake suke wajen".split()  # noqa: SIM905
    ),
    Language.YORUBA: frozenset(
        "níbo nibo mo ti lè le wọlé ṣí si ojú oju ìwé iwe fún fun kí ki ni gbọdọ gbọdọ̀ ṣe "
        "ìgbésẹ̀ igbesẹ ṣàlàyé salaye igba melo báwo bawo kini ilana ọsẹ ose mi abi ọjọ ojo wo "
        "àwọn awon máa maa jẹ je lọ lo hàn han".split()  # noqa: SIM905
    ),
    Language.IGBO: frozenset(
        "ebee ebe ka ga abanye banye meghee peeji gịnị gini kọwaa kowaa kedu iwu maka ole achọrọ "
        "achoro enwere nke ahụ ahu ọhụrụ ohuru bụ bu ọ bụrụ o bụrụ bụrụ gi gị mee ndị ndi enye "
        "efu nọ no".split()  # noqa: SIM905
    ),
}

_PORTAL_TERMS = {
    PortalAction.LOGIN: (
        "sign in", "signin", "login", "log in", "wọlé", "wole", "shiga", "abanye", "banye",
    ),
    PortalAction.TRACK_SELECTION: (
        "track selection", "select track", "track page", "zaben track", "zaɓen track",
    ),
    PortalAction.PROFILE: ("profile", "profaịlụ"),
    PortalAction.SUPPORT: ("support", "contact"),
    PortalAction.COURSES: (
        "courses", "course page", "my courses", "see my courses", "hụ courses",
    ),
    PortalAction.REGISTER: (
        "registration page", "register page", "registration", "register", "rajistar",
        "ìforúkọsílẹ̀", "iforukosile", "ndebanye aha",
    ),
    PortalAction.NCAIR_HOME: (
        "ncair website", "official ncair website", "gidan yanar gizon ncair",
        "website ncair", "ncair home",
    ),
    PortalAction.MAIN: (
        "main ncair lms", "main lms", "lms home", "home page", "isi peeji ncair lms",
        "ojú ìwé àkọ́kọ́ ncair lms", "oju iwe akoko ncair lms", "shafin ncair lms",
    ),
}

_NAVIGATION_ACTION_CUES = (
    "open",
    "show me",
    "take me",
    "go to",
    "visit",
    "bude",
    "buɗe",
    "kai ni",
    "nuna min",
    "ṣí",
    "si oju iwe",
    "mú mi lọ",
    "mu mi lo",
    "meghee",
)

_LOGIN_NAVIGATION_PATTERNS = (
    r"\bwhere\s+(?:do|can|should)\s+i\s+(?:sign\s*in|log\s*in|login)\b",
    r"\bina\s+zan\s+shiga\b",
    r"\bnibo\s+ni\s+mo\s+ti\s+le\s+wole\b",
    r"\bebee\s+ka\s+m\s+ga[-\s]?abanye\b",
)

_STEP_CUES = ("step", "mataki", "ìgbésẹ̀", "igbese")


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

        kwargs = {
            "device_map": self._settings.natlas_device,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
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
            model.eval()
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
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text.strip(chr(96)).strip()
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


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^\w]+", " ", without_marks, flags=re.UNICODE).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", text.casefold()), re.UNICODE)


def _language_hint(question: str) -> tuple[Language | None, int]:
    words = _words(question)
    ownership: dict[str, list[Language]] = {}
    for language, lexicon in _LANGUAGE_LEXICON.items():
        for word in lexicon:
            ownership.setdefault(word, []).append(language)

    scores = dict.fromkeys(Language, 0)
    for word in words:
        owners = ownership.get(word, [])
        if len(owners) == 1:
            scores[owners[0]] += 1

    normalized = unicodedata.normalize("NFC", question.casefold())
    if re.search(r"[ɓɗƙƴ]", normalized):
        scores[Language.HAUSA] += 5
    if re.search(r"[ṣẹ]", normalized):
        scores[Language.YORUBA] += 5
    if re.search(r"[ịụṅ]", normalized):
        scores[Language.IGBO] += 5
    if re.search(r"[àáèéìíòóùú]", normalized) and scores[Language.YORUBA] > 0:
        scores[Language.YORUBA] += 1
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] == 0:
        return None, 0
    margin = ranked[0][1] - ranked[1][1]
    if margin <= 0:
        return None, margin
    return ranked[0][0], margin


def _explicit_step(question: str) -> int | None:
    folded = _fold(question)
    if not any(re.search(rf"\b{re.escape(_fold(cue))}\b", folded) for cue in _STEP_CUES):
        return None
    match = re.search(r"\b([1-4])\b", folded)
    if match:
        return int(match.group(1))
    number_words = {
        "one": 1, "first": 1, "two": 2, "second": 2,
        "three": 3, "third": 3, "four": 4, "fourth": 4,
    }
    for word, number in number_words.items():
        if re.search(rf"\b{word}\b", folded):
            return number
    return None


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = re.escape(_fold(phrase)).replace(r"\ ", r"\s+")
    return re.search(rf"(?:^|\s){normalized}(?:\s|$)", text) is not None


def _has_navigation_intent(question: str) -> bool:
    folded = _fold(question)
    if any(_contains_phrase(folded, cue) for cue in _NAVIGATION_ACTION_CUES):
        return True
    if re.search(r"\b(?:need|want)\b.*\b(?:page|link|url|website|portal)\b", folded):
        return True
    if re.search(
        r"\bina\s+son\b.*\b(?:shafi|website|gidan\s+yanar\s+gizon?)\b",
        folded,
    ):
        return True
    return any(re.search(pattern, folded) for pattern in _LOGIN_NAVIGATION_PATTERNS)


def _explicit_portal_action(question: str) -> PortalAction | None:
    folded = f" {_fold(question)} "
    for action, terms in _PORTAL_TERMS.items():
        for term in terms:
            if f" {_fold(term)} " in folded:
                return action
    return None


def _parse_language(raw: str) -> Language:
    try:
        return Language(str(_json_object(raw)["language"]).strip().lower())
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidModelOutputError("N-ATLaS returned an invalid language label.") from exc


def _parse_tool(raw: str) -> ToolName:
    payload = _json_object(raw)
    tool_values = {tool.value for tool in ToolName}
    candidates: list[str] = []
    if "tool" in payload:
        candidates.append(str(payload["tool"]))
    if "action" in payload and str(payload["action"]) in tool_values:
        candidates.append(str(payload["action"]))
    candidates.extend(key for key in payload if key in tool_values)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise InvalidModelOutputError("N-ATLaS routing output must identify one documented tool.")
    try:
        return ToolName(candidates[0])
    except ValueError as exc:
        raise InvalidModelOutputError("N-ATLaS returned an unknown tool.") from exc


def _parse_portal_action(raw: str) -> PortalAction:
    payload = _json_object(raw)
    value = payload.get("action")
    if isinstance(value, dict):
        value = value.get("action")
    try:
        action = PortalAction(str(value).strip().lower())
    except ValueError as exc:
        raise InvalidModelOutputError("N-ATLaS returned an invalid portal action.") from exc
    return PortalAction.LOGIN if action is PortalAction.SIGNIN else action


def _parse_supported(raw: str) -> bool:
    supported = _json_object(raw).get("supported")
    if not isinstance(supported, bool):
        raise InvalidModelOutputError("N-ATLaS evidence verdict must contain a boolean.")
    return supported


class NatlasLanguageDetector:
    def __init__(self, client: NatlasTextClient):
        self._client = client

    def _model_detect(self, question: str) -> Language:
        messages = [
            {"role": "system", "content": LANGUAGE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        raw = self._client.generate(messages, max_new_tokens=40)
        try:
            return _parse_language(raw)
        except InvalidModelOutputError as first_error:
            LOGGER.warning("natlas_invalid_language_output raw=%r", raw[:1000])
            retry = self._client.generate(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Return only the required JSON with one language value: "
                            "english, hausa, yoruba, or igbo."
                        ),
                    },
                ],
                max_new_tokens=40,
            )
            try:
                return _parse_language(retry)
            except InvalidModelOutputError as retry_error:
                raise retry_error from first_error

    def detect(self, question: str) -> Language:
        hint, margin = _language_hint(question)
        if hint is not None and margin > 0:
            LOGGER.debug("language_lexical_hint language=%s margin=%s", hint.value, margin)
            return hint
        return self._model_detect(question)


class NatlasRouter:
    def __init__(
        self,
        client: NatlasTextClient,
        language_detector: NatlasLanguageDetector | None = None,
    ):
        self._client = client
        self._language_detector = language_detector or NatlasLanguageDetector(client)

    def _model_tool(self, question: str, *, language: Language) -> ToolName:
        messages = [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Input language: {language.value}\nUser request:\n{question}",
            },
        ]
        raw = self._client.generate(messages, max_new_tokens=70)
        try:
            return _parse_tool(raw)
        except InvalidModelOutputError as first_error:
            LOGGER.warning("natlas_invalid_tool_output raw=%r", raw[:1000])
            retry = self._client.generate(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Return only one valid JSON tool choice using exactly one of "
                            "get_portal_link, get_step_guidance, or search_ncair_knowledge_base."
                        ),
                    },
                ],
                max_new_tokens=70,
            )
            try:
                return _parse_tool(retry)
            except InvalidModelOutputError as retry_error:
                raise retry_error from first_error

    def _model_portal_action(self, question: str) -> PortalAction:
        messages = [
            {"role": "system", "content": PORTAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        raw = self._client.generate(messages, max_new_tokens=50)
        try:
            return _parse_portal_action(raw)
        except InvalidModelOutputError as first_error:
            LOGGER.warning("natlas_invalid_portal_action raw=%r", raw[:1000])
            retry = self._client.generate(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Return only one valid JSON action using an exact allowed "
                            "action value."
                        ),
                    },
                ],
                max_new_tokens=50,
            )
            try:
                return _parse_portal_action(retry)
            except InvalidModelOutputError as retry_error:
                raise retry_error from first_error

    @staticmethod
    def _knowledge(language: Language, question: str) -> RoutingDecision:
        return RoutingDecision(
            language=language,
            tool=ToolName.KNOWLEDGE,
            retrieval_query=question.strip(),
            status=RoutingStatus.MODEL,
        ).validate()

    def route(self, question: str) -> RoutingDecision:
        language = self._language_detector.detect(question)

        # Specialized tools have explicit contracts. Resolve those contracts before asking the
        # model so malformed model output cannot break an otherwise unambiguous request.
        step = _explicit_step(question)
        if step is not None:
            return RoutingDecision(
                language=language,
                tool=ToolName.STEP_GUIDANCE,
                step=step,
                status=RoutingStatus.MODEL,
            ).validate()

        navigation_intent = _has_navigation_intent(question)
        portal_action = _explicit_portal_action(question) if navigation_intent else None
        if navigation_intent and portal_action is not None:
            return RoutingDecision(
                language=language,
                tool=ToolName.PORTAL_LINK,
                action=portal_action,
                status=RoutingStatus.MODEL,
            ).validate()

        # For ambiguous requests, N-ATLaS remains the semantic router. The deterministic contract
        # validator may only make a specialized decision stricter, never invent one.
        try:
            model_tool = self._model_tool(question, language=language)
        except InvalidModelOutputError:
            LOGGER.warning("natlas_tool_fallback_to_knowledge")
            return self._knowledge(language, question)

        if model_tool is ToolName.KNOWLEDGE:
            return self._knowledge(language, question)
        if model_tool is ToolName.STEP_GUIDANCE:
            return self._knowledge(language, question)
        if not navigation_intent:
            return self._knowledge(language, question)

        try:
            action = self._model_portal_action(question)
        except InvalidModelOutputError:
            return self._knowledge(language, question)

        return RoutingDecision(
            language=language,
            tool=ToolName.PORTAL_LINK,
            action=action,
            status=RoutingStatus.MODEL,
        ).validate()


class NatlasEvidenceVerifier:
    def __init__(self, client: NatlasTextClient):
        self._client = client

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
            return _parse_supported(raw)
        except InvalidModelOutputError as first_error:
            LOGGER.warning("natlas_invalid_evidence_output raw=%r", raw[:1000])
            retry = self._client.generate(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": 'Return only {"supported":true} or {"supported":false}.',
                    },
                ],
                max_new_tokens=40,
            )
            try:
                return _parse_supported(retry)
            except InvalidModelOutputError:
                LOGGER.warning("natlas_invalid_evidence_retry raw=%r", retry[:1000])
                return False


class NatlasGroundedAnswerer:
    def __init__(self, client: NatlasTextClient, settings: Settings):
        self._client = client
        self._settings = settings

    def answer(self, *, question: str, language: Language, evidence: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the NCAIR LMS Assistant. Answer only from the supplied verified "
                    "official evidence. Do not invent dates, contacts, policies, URLs, or "
                    f"requirements. Answer in {language.value}. Preserve every URL exactly as "
                    "written. Be concise."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nVerified official evidence:\n{evidence}",
            },
        ]
        answer = self._client.generate(
            messages,
            max_new_tokens=self._settings.natlas_max_new_tokens,
        ).strip()
        if not answer:
            raise InvalidModelOutputError("N-ATLaS returned an empty answer.")
        return answer
