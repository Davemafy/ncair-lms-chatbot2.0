from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The live ZeroGPU demo intentionally uses the curated official text knowledge
# source only. The canonical repository benchmark still uses the complete data/
# directory, including the onboarding manual.
SPACE_ROOT = Path(__file__).resolve().parent
DATA_DIR = SPACE_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

KNOWLEDGE_URL = (
    "https://raw.githubusercontent.com/Davemafy/ncair-lms-chatbot2.0/"
    "c1e4f33850a1b225ab8afa8fade48fa641a9f856/data/ncair_knowledge_base.txt"
)
KNOWLEDGE_PATH = DATA_DIR / "ncair_knowledge_base.txt"
if not KNOWLEDGE_PATH.exists():
    urlretrieve(KNOWLEDGE_URL, KNOWLEDGE_PATH)

os.environ["NCAIR_DATA_DIR"] = str(DATA_DIR)
os.environ.setdefault("NCAIR_DEFAULT_VERSION", "v2")
os.environ.setdefault("NATLAS_MODEL", "NCAIR1/N-ATLaS")
os.environ.setdefault("TOP_K", "3")

from ncair_lms.config import Settings
from ncair_lms.natlas import NatlasGroundedAnswerer, NatlasRouter
from ncair_lms.service import AssistantService

SETTINGS = Settings.from_env()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is required. Add it as a Hugging Face Space secret after "
        "accepting the NCAIR1/N-ATLaS model access conditions."
    )


class ZeroGpuNatlasClient:
    """N-ATLaS adapter whose model is resident on ZeroGPU CUDA emulation."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            SETTINGS.natlas_model,
            token=HF_TOKEN,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            SETTINGS.natlas_model,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=False,
            date_string=datetime.now().strftime("%d %b %Y"),
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")
        prompt_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.05,
            )

        generated = output[0][prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


NATLAS = ZeroGpuNatlasClient()
ROUTER = NatlasRouter(NATLAS)
ANSWERER = NatlasGroundedAnswerer(NATLAS, SETTINGS)
SERVICE = AssistantService(
    "v2",
    settings=SETTINGS,
    router=ROUTER,
    answerer=ANSWERER,
)


@spaces.GPU(duration=120)
def chat(message: str) -> dict:
    """Run one complete N-ATLaS V2 turn and return stable API metadata."""

    message = message.strip()
    if not message:
        return {
            "answer": "Please enter a question.",
            "language": "english",
            "tool": "search_ncair_knowledge_base",
            "sources": [],
        }
    return SERVICE.chat(message).as_dict()


demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(
        label="Question",
        placeholder="Ask in English, Hausa, Yoruba or Igbo",
        lines=2,
    ),
    outputs=gr.JSON(label="NCAIR LMS Assistant 2.0"),
    title="NCAIR LMS Chatbot 2.0 — N-ATLaS",
    description=(
        "Text-only multilingual demo using NCAIR1/N-ATLaS for semantic routing "
        "and grounded answers over official NCAIR LMS evidence."
    ),
    api_name="chat",
)

if __name__ == "__main__":
    demo.launch()
