from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from .config import Settings
from .errors import InvalidModelOutputError, ModelUnavailableError
from .models import Language, PortalAction, RoutingDecision, RoutingStatus, ToolName

LOGGER = logging.getLogger(__name__)


LANGUAGE_SYSTEM_PROMPT = """Classify the language carrying the grammar of the user request.
Choose exactly one option:
A = English
B = Hausa
C = Yoruba
D = Igbo

Ignore names, URLs, acronyms, course names, and borrowed English technical words when the
surrounding sentence grammar belongs to another language. For code-switched text, classify
the language carrying most of the sentence structure.
"""


ROUTE_SYSTEM_PROMPT = """Classify the requested outcome for the NCAIR LMS assistant.
Choose exactly one option:

A = LMS main/home page
B = LMS sign-in/login page
C = official NCAIR website home page
D = LMS registration page
E = intern profile page
F = courses page
G = track-selection page
H = NCAIR support/contact page
I = onboarding step 1
J = onboarding step 2
K = onboarding step 3
L = onboarding step 4
M = knowledge question

Use A-H only when the user wants that page, link, URL, or destination itself.
Use I-L only when the user explicitly asks about that numbered onboarding step.
Use M when the user wants information: a fact, policy, requirement, rule, limit, schedule,
explanation, troubleshooting answer, verification of a claim, or information that may not be
documented. A request remains M when it mentions a portal topic but asks for information about
that topic instead of asking to open the page.
"""


class RouteLabel(StrEnum):
    LMS_HOME = "A"
    SIGN_IN = "B"
    NCAIR_HOME = "C"
    REGISTER = "D"
    PROFILE = "E"
    COURSES = "F"
    TRACK_SELECTION = "G"
    SUPPORT = "H"
    STEP_1 = "I"
    STEP_2 = "J"
    STEP_3 = "K"
    STEP_4 = "L"
    KNOWLEDGE = "M"


_LANGUAGE_BY_CHOICE = {
    "A": Language.ENGLISH,
    "B": Language.HAUSA,
    "C": Language.YORUBA,
    "D": Language.IGBO,
}

_ROUTE_CHOICES = tuple(label.value for label in RouteLabel)


class NatlasClient(Protocol):
    def generate(self, messages: Sequence[dict[str, str]], *, max_new_tokens: int) -> str: ...

    def choose(self, messages: Sequence[dict[str, str]], choices: Sequence[str]) -> str: ...


class LocalNatlasClient:
    """Lazy local adapter for NCAIR1/N-ATLaS.

    Generation is used only for the final grounded answer. Language and routing are closed-set
    classifications computed from candidate-token likelihoods, so invalid JSON/tool outputs are
    impossible.
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
        """Return the highest-likelihood allowed continuation without free-form generation."""
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        if not choices:
            raise ValueError("choices must not be empty")

        try:
            import torch

            prompt = self._chat_prompt(messages)
            prompt_ids = self._tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].tolist()
            if not prompt_ids:
                raise InvalidModelOutputError("N-ATLaS classification prompt tokenized to empty.")

            choice_ids = [
                self._tokenizer(choice, add_special_tokens=False)["input_ids"] for choice in choices
            ]
            if any(not token_ids for token_ids in choice_ids):
                raise InvalidModelOutputError("A classification choice tokenized to empty.")

            pad_id = self._tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self._tokenizer.eos_token_id
            if pad_id is None:
                raise InvalidModelOutputError("N-ATLaS tokenizer has no pad/eos token.")

            max_length = len(prompt_ids) + max(len(token_ids) for token_ids in choice_ids)
            batch = torch.full(
                (len(choices), max_length),
                pad_id,
                dtype=torch.long,
            )
            attention = torch.zeros_like(batch)

            for row, token_ids in enumerate(choice_ids):
                sequence = prompt_ids + token_ids
                batch[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                attention[row, : len(sequence)] = 1

            model_device = next(self._model.parameters()).device
            batch = batch.to(model_device)
            attention = attention.to(model_device)

            with torch.no_grad():
                logits = self._model(input_ids=batch, attention_mask=attention).logits.float()
                log_probs = torch.log_softmax(logits, dim=-1)

            scores = []
            prompt_length = len(prompt_ids)
            for row, token_ids in enumerate(choice_ids):
                token_scores = [
                    log_probs[row, prompt_length + offset - 1, token_id].item()
                    for offset, token_id in enumerate(token_ids)
                ]
                scores.append(sum(token_scores) / len(token_scores))

            best_index = max(range(len(scores)), key=scores.__getitem__)
            LOGGER.debug(
                "natlas_closed_set_scores choices=%s scores=%s selected=%s",
                list(choices),
                [round(score, 4) for score in scores],
                choices[best_index],
            )
            return choices[best_index]
        except (InvalidModelOutputError, ValueError):
            raise
        except Exception as exc:
            raise ModelUnavailableError("N-ATLaS closed-set classification failed.") from exc


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
            return _LANGUAGE_BY_CHOICE[choice]
        except KeyError as exc:
            raise InvalidModelOutputError(
                f"N-ATLaS returned an unknown language choice {choice!r}."
            ) from exc


def _route_decision(label: RouteLabel, *, language: Language, question: str) -> RoutingDecision:
    portal = {
        RouteLabel.LMS_HOME: PortalAction.MAIN,
        RouteLabel.SIGN_IN: PortalAction.LOGIN,
        RouteLabel.NCAIR_HOME: PortalAction.NCAIR_HOME,
        RouteLabel.REGISTER: PortalAction.REGISTER,
        RouteLabel.PROFILE: PortalAction.PROFILE,
        RouteLabel.COURSES: PortalAction.COURSES,
        RouteLabel.TRACK_SELECTION: PortalAction.TRACK_SELECTION,
        RouteLabel.SUPPORT: PortalAction.SUPPORT,
    }
    steps = {
        RouteLabel.STEP_1: 1,
        RouteLabel.STEP_2: 2,
        RouteLabel.STEP_3: 3,
        RouteLabel.STEP_4: 4,
    }

    if label in portal:
        return RoutingDecision(
            language=language,
            tool=ToolName.PORTAL_LINK,
            action=portal[label],
            status=RoutingStatus.MODEL,
        ).validate()

    if label in steps:
        return RoutingDecision(
            language=language,
            tool=ToolName.STEP_GUIDANCE,
            step=steps[label],
            status=RoutingStatus.MODEL,
        ).validate()

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
        choice = self._client.choose(
            [
                {"role": "system", "content": ROUTE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Detected language: {language.value}\nRequest: {question}",
                },
            ],
            _ROUTE_CHOICES,
        )
        try:
            label = RouteLabel(choice)
        except ValueError as exc:
            raise InvalidModelOutputError(
                f"N-ATLaS returned an unknown route choice {choice!r}."
            ) from exc
        return _route_decision(label, language=language, question=question)


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
