from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from .config import Settings
from .errors import InvalidModelOutputError, ModelUnavailableError
from .models import Language, PortalAction, RoutingDecision, RoutingStatus, ToolName


LOGGER = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the semantic router for the NCAIR LMS assistant.
Return exactly one JSON object and no prose.

Supported languages:
- english
- hausa
- yoruba
- igbo

Allowed tools:
1. get_portal_link
   arguments: {"action": one of the documented portal actions}
2. get_step_guidance
   arguments: {"step": 1, 2, 3, or 4}
3. search_ncair_knowledge_base
   arguments: {"query": "<concise English retrieval query>"}

Schema:
{
  "language": "english|hausa|yoruba|igbo",
  "tool": "get_portal_link|get_step_guidance|search_ncair_knowledge_base",
  "arguments": {...}
}

Rules:
- Choose exactly one tool.
- Understand paraphrases semantically, not by literal keywords.
- For Hausa, Yoruba, or Igbo knowledge questions, translate the retrieval query into English.
- Use the knowledge-base tool for policies, programme facts, troubleshooting, and unsupported facts.
- Do not invent tools, actions, or step numbers.
"""


class NatlasTextClient(Protocol):
    def generate(self, messages: Sequence[dict[str, str]], *, max_new_tokens: int) -> str:
        ...


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
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelUnavailableError(
                "N-ATLaS requires transformers, accelerate, and torch. "
                "Install the runtime dependencies first."
            ) from exc

        kwargs = {"device_map": self._settings.natlas_device, "torch_dtype": "auto"}
        if self._settings.hf_token:
            kwargs["token"] = self._settings.hf_token

        LOGGER.info("loading_natlas model=%s", self._settings.natlas_model)
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


def _json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    start = text.find("{")
    if start < 0:
        raise InvalidModelOutputError("N-ATLaS did not return a JSON object.")

    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise InvalidModelOutputError("N-ATLaS returned invalid JSON.") from exc

    if not isinstance(value, dict):
        raise InvalidModelOutputError("N-ATLaS routing output must be a JSON object.")
    return value


class NatlasRouter:
    def __init__(self, client: NatlasTextClient):
        self._client = client

    def route(self, question: str) -> RoutingDecision:
        raw = self._client.generate(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_new_tokens=180,
        )
        payload = _json_object(raw)

        try:
            language = Language(str(payload["language"]).lower())
            tool = ToolName(str(payload["tool"]))
            arguments = payload["arguments"]
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidModelOutputError("N-ATLaS routing output has an invalid schema.") from exc

        if not isinstance(arguments, dict):
            raise InvalidModelOutputError("N-ATLaS tool arguments must be an object.")

        if tool is ToolName.PORTAL_LINK:
            try:
                action = PortalAction(str(arguments["action"]).lower())
            except (KeyError, ValueError) as exc:
                raise InvalidModelOutputError("Invalid get_portal_link action.") from exc
            return RoutingDecision(
                language=language,
                tool=tool,
                action=action,
                status=RoutingStatus.MODEL,
            ).validate()

        if tool is ToolName.STEP_GUIDANCE:
            step = arguments.get("step")
            if isinstance(step, bool) or not isinstance(step, int):
                raise InvalidModelOutputError("get_step_guidance.step must be an integer.")
            return RoutingDecision(
                language=language,
                tool=tool,
                step=step,
                status=RoutingStatus.MODEL,
            ).validate()

        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise InvalidModelOutputError(
                "search_ncair_knowledge_base.query must be a non-empty string."
            )
        return RoutingDecision(
            language=language,
            tool=tool,
            retrieval_query=query.strip(),
            status=RoutingStatus.MODEL,
        ).validate()


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
