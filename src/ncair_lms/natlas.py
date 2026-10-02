from __future__ import annotations

import logging
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from .config import Settings
from .errors import InvalidModelOutputError, ModelUnavailableError
from .models import Language, PortalAction, RoutingDecision, RoutingStatus, ToolName

LOGGER = logging.getLogger(__name__)


LANGUAGE_SYSTEM_PROMPT = """Identify the language carrying the grammar of the user request.
Return exactly one label: English, Hausa, Yoruba, or Igbo.

Ignore names, URLs, acronyms, course names, and borrowed English technical words. For
code-switched text, choose the language carrying most of the sentence structure. Do not
translate the request and do not explain your choice.
"""


OUTCOME_SYSTEM_PROMPT = """Classify what outcome the user wants from the NCAIR LMS assistant.
Return exactly one label: navigation, step guidance, or knowledge answer.

navigation = the user wants a page, link, URL, or destination opened or shown.
step guidance = the user explicitly wants one numbered onboarding step.
knowledge answer = the user wants information: a fact, policy, requirement, rule, limit,
schedule, explanation, troubleshooting answer, or verification of a claim.

Classify the requested outcome, not topic words. A factual question about login, registration,
courses, profile, support, or any other portal topic is still knowledge answer when the user
wants information rather than the page itself.
"""


PAGE_SYSTEM_PROMPT = """The user wants navigation. Identify the requested destination.
Return exactly one label from this set:

LMS home
LMS sign-in
NCAIR website
LMS registration
intern profile
courses
track selection
NCAIR support

Choose the destination itself. Do not explain your choice.
"""


STEP_SYSTEM_PROMPT = """The user wants numbered onboarding guidance.
Return exactly one label: step 1, step 2, step 3, or step 4.
Do not explain your choice.
"""


class OutcomeKind(StrEnum):
    NAVIGATION = "navigation"
    STEP_GUIDANCE = "step guidance"
    KNOWLEDGE = "knowledge answer"


class PageLabel(StrEnum):
    LMS_HOME = "LMS home"
    SIGN_IN = "LMS sign-in"
    NCAIR_HOME = "NCAIR website"
    REGISTER = "LMS registration"
    PROFILE = "intern profile"
    COURSES = "courses"
    TRACK_SELECTION = "track selection"
    SUPPORT = "NCAIR support"


class StepLabel(StrEnum):
    STEP_1 = "step 1"
    STEP_2 = "step 2"
    STEP_3 = "step 3"
    STEP_4 = "step 4"


_LANGUAGE_BY_CHOICE = {
    "English": Language.ENGLISH,
    "Hausa": Language.HAUSA,
    "Yoruba": Language.YORUBA,
    "Igbo": Language.IGBO,
}

_PAGE_ACTIONS = {
    PageLabel.LMS_HOME: PortalAction.MAIN,
    PageLabel.SIGN_IN: PortalAction.LOGIN,
    PageLabel.NCAIR_HOME: PortalAction.NCAIR_HOME,
    PageLabel.REGISTER: PortalAction.REGISTER,
    PageLabel.PROFILE: PortalAction.PROFILE,
    PageLabel.COURSES: PortalAction.COURSES,
    PageLabel.TRACK_SELECTION: PortalAction.TRACK_SELECTION,
    PageLabel.SUPPORT: PortalAction.SUPPORT,
}

_STEP_NUMBERS = {
    StepLabel.STEP_1: 1,
    StepLabel.STEP_2: 2,
    StepLabel.STEP_3: 3,
    StepLabel.STEP_4: 4,
}


class NatlasClient(Protocol):
    def generate(self, messages: Sequence[dict[str, str]], *, max_new_tokens: int) -> str: ...

    def choose(self, messages: Sequence[dict[str, str]], choices: Sequence[str]) -> str: ...


class LocalNatlasClient:
    """Lazy local adapter for NCAIR1/N-ATLaS.

    Final answers use ordinary deterministic generation. Classification uses constrained
    decoding over meaningful labels, so the model cannot emit invalid language, intent, page,
    or step values and does not have to compare opaque token codes.
    """

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

    def _chat_prompt(self, messages: Sequence[dict[str, str]]) -> str:
        assert self._tokenizer is not None
        return self._tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=False,
            date_string=datetime.now().strftime("%d %b %Y"),
        )

    def generate(self, messages: Sequence[dict[str, str]], *, max_new_tokens: int) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        try:
            prompt = self._chat_prompt(messages)
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

    def choose(self, messages: Sequence[dict[str, str]], choices: Sequence[str]) -> str:
        """Generate exactly one allowed semantic label using a token-prefix trie."""
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        normalized_choices = tuple(choice.strip() for choice in choices)
        if not normalized_choices or any(not choice for choice in normalized_choices):
            raise ValueError("classification choices must be non-empty")
        if len(set(normalized_choices)) != len(normalized_choices):
            raise ValueError("classification choices must be unique")

        try:
            prompt = self._chat_prompt(messages)
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_length = inputs["input_ids"].shape[-1]
            model_device = next(self._model.parameters()).device
            inputs = {name: value.to(model_device) for name, value in inputs.items()}

            # Labels are explicit natural-language continuations. Prefixing one space makes
            # continuation tokenization stable across the allowed set.
            candidate_ids = [
                self._tokenizer(f" {choice}", add_special_tokens=False)["input_ids"]
                for choice in normalized_choices
            ]
            if any(not token_ids for token_ids in candidate_ids):
                raise InvalidModelOutputError("A classification choice tokenized to empty.")

            stop_ids = {self._tokenizer.eos_token_id}
            eot_id = self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_id, int) and eot_id >= 0:
                stop_ids.add(eot_id)
            stop_ids.discard(None)
            if not stop_ids:
                raise InvalidModelOutputError("N-ATLaS tokenizer has no usable stop token.")

            def allowed_tokens(_batch_id, input_ids):
                generated = input_ids[prompt_length:].tolist()
                allowed: set[int] = set()

                for token_ids in candidate_ids:
                    if token_ids[: len(generated)] != generated:
                        continue
                    if len(generated) < len(token_ids):
                        allowed.add(token_ids[len(generated)])
                    else:
                        allowed.update(stop_ids)

                if not allowed:
                    allowed.update(stop_ids)
                return sorted(allowed)

            max_new_tokens = max(len(token_ids) for token_ids in candidate_ids) + 1
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                prefix_allowed_tokens_fn=allowed_tokens,
                eos_token_id=sorted(stop_ids),
                pad_token_id=self._tokenizer.eos_token_id,
            )

            generated_ids = output[0][prompt_length:].tolist()
            while generated_ids and generated_ids[-1] in stop_ids:
                generated_ids.pop()

            matches = [
                choice
                for choice, token_ids in zip(
                    normalized_choices,
                    candidate_ids,
                    strict=True,
                )
                if generated_ids == token_ids
            ]
            if len(matches) != 1:
                raw = self._tokenizer.decode(
                    output[0][prompt_length:],
                    skip_special_tokens=True,
                )
                raise InvalidModelOutputError(
                    f"N-ATLaS constrained classification did not resolve uniquely: {raw!r}"
                )

            LOGGER.debug("natlas_constrained_choice selected=%s", matches[0])
            return matches[0]
        except (InvalidModelOutputError, ValueError):
            raise
        except Exception as exc:
            raise ModelUnavailableError("N-ATLaS constrained classification failed.") from exc


def _orthographic_language_hint(question: str) -> Language | None:
    """Return a language only when distinctive orthography gives one unambiguous signal."""
    text = unicodedata.normalize("NFC", question.casefold())
    marker_sets = {
        Language.HAUSA: set("ɓɗƙƴ".casefold()),
        Language.YORUBA: {"ṣ", "ẹ"},
        Language.IGBO: {"ị", "ụ", "ṅ"},
    }
    matches = [
        language
        for language, markers in marker_sets.items()
        if any(marker in text for marker in markers)
    ]
    return matches[0] if len(matches) == 1 else None


class NatlasLanguageDetector:
    def __init__(self, client: NatlasClient):
        self._client = client

    def detect(self, question: str) -> Language:
        choice = self._client.choose(
            [
                {"role": "system", "content": LANGUAGE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tuple(_LANGUAGE_BY_CHOICE),
        )
        try:
            model_language = _LANGUAGE_BY_CHOICE[choice]
        except KeyError as exc:
            raise InvalidModelOutputError(
                f"N-ATLaS returned an unknown language choice {choice!r}."
            ) from exc

        orthographic_hint = _orthographic_language_hint(question)
        if orthographic_hint is not None and orthographic_hint is not model_language:
            LOGGER.info(
                "language_orthography_override model=%s orthography=%s",
                model_language.value,
                orthographic_hint.value,
            )
            return orthographic_hint
        return model_language


def _knowledge_decision(*, language: Language, question: str) -> RoutingDecision:
    return RoutingDecision(
        language=language,
        tool=ToolName.KNOWLEDGE,
        retrieval_query=question.strip(),
        status=RoutingStatus.MODEL,
    ).validate()


class NatlasRouter:
    def __init__(
        self,
        client: NatlasClient,
        language_detector: NatlasLanguageDetector | None = None,
    ):
        self._client = client
        self._language_detector = language_detector or NatlasLanguageDetector(client)

    def route(self, question: str) -> RoutingDecision:
        language = self._language_detector.detect(question)

        outcome_choice = self._client.choose(
            [
                {"role": "system", "content": OUTCOME_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tuple(kind.value for kind in OutcomeKind),
        )
        try:
            outcome = OutcomeKind(outcome_choice)
        except ValueError as exc:
            raise InvalidModelOutputError(
                f"N-ATLaS returned an unknown outcome choice {outcome_choice!r}."
            ) from exc

        if outcome is OutcomeKind.KNOWLEDGE:
            return _knowledge_decision(language=language, question=question)

        if outcome is OutcomeKind.NAVIGATION:
            page_choice = self._client.choose(
                [
                    {"role": "system", "content": PAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                tuple(label.value for label in PageLabel),
            )
            try:
                page = PageLabel(page_choice)
            except ValueError as exc:
                raise InvalidModelOutputError(
                    f"N-ATLaS returned an unknown page choice {page_choice!r}."
                ) from exc
            return RoutingDecision(
                language=language,
                tool=ToolName.PORTAL_LINK,
                action=_PAGE_ACTIONS[page],
                status=RoutingStatus.MODEL,
            ).validate()

        step_choice = self._client.choose(
            [
                {"role": "system", "content": STEP_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tuple(label.value for label in StepLabel),
        )
        try:
            step = StepLabel(step_choice)
        except ValueError as exc:
            raise InvalidModelOutputError(
                f"N-ATLaS returned an unknown step choice {step_choice!r}."
            ) from exc
        return RoutingDecision(
            language=language,
            tool=ToolName.STEP_GUIDANCE,
            step=_STEP_NUMBERS[step],
            status=RoutingStatus.MODEL,
        ).validate()


class NatlasGroundedAnswerer:
    def __init__(self, client: NatlasClient, settings: Settings):
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
