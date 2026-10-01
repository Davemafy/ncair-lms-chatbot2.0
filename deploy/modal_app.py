from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "ncair-lms-natlas-v2"
MODEL_ID = "NCAIR1/N-ATLaS"
HF_CACHE_PATH = "/cache/huggingface"

ROOT = Path(__file__).resolve().parents[1]

hf_cache = modal.Volume.from_name("ncair-lms-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name(
    "huggingface-secret",
    required_keys=["HF_TOKEN"],
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("poppler-utils", "tesseract-ocr")
    .uv_pip_install(
        "fastapi>=0.116.0",
        "pydantic>=2.7.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "accelerate>=1.0.0",
        "faiss-cpu>=1.7.4",
        "pdf2image>=1.17.0",
        "pytesseract>=0.3.10",
        "sentence-transformers>=3.0.0",
        "torch>=2.8.0",
        "transformers>=4.45.0",
        "uvicorn[standard]>=0.35.0",
        "huggingface-hub>=0.35.0",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_HUB_CACHE": f"{HF_CACHE_PATH}/hub",
            "SENTENCE_TRANSFORMERS_HOME": f"{HF_CACHE_PATH}/sentence-transformers",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "NATLAS_MODEL": MODEL_ID,
            "NATLAS_DEVICE": "auto",
            "NCAIR_DEFAULT_VERSION": "v2",
            "NCAIR_DATA_DIR": "/data",
            "TOP_K": "3",
            "MIN_RETRIEVAL_SCORE": "0.30",
        }
    )
    .add_local_dir(ROOT / "src" / "ncair_lms", "/root/ncair_lms")
    .add_local_dir(ROOT / "data", "/data")
    .add_local_dir(ROOT / "web", "/web")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={"/cache": hf_cache},
    timeout=30 * 60,
)
def download_models() -> None:
    """Populate the persistent Hugging Face cache without using a GPU."""

    from huggingface_hub import snapshot_download

    token = os.environ["HF_TOKEN"]

    snapshot_download(
        repo_id=MODEL_ID,
        token=token,
        cache_dir=f"{HF_CACHE_PATH}/hub",
    )
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=f"{HF_CACHE_PATH}/hub",
    )

    hf_cache.commit()


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    volumes={"/cache": hf_cache},
    timeout=15 * 60,
    max_containers=1,
    scaledown_window=30,
)
@modal.asgi_app()
def web():
    """Serve the canonical V2 FastAPI app on one cost-capped L4 container."""

    from fastapi.middleware.cors import CORSMiddleware

    from ncair_lms.api import app as api

    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://tbotv1.vercel.app",
            "https://ncair-lms-chatbotv1.vercel.app",
            "http://localhost:8000",
            "http://localhost:4173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    return api
