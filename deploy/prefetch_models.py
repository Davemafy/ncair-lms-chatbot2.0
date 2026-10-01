from __future__ import annotations

import os

from huggingface_hub import snapshot_download

MODEL_ID = os.getenv("NATLAS_MODEL", "NCAIR1/N-ATLaS")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise SystemExit(
        "HF_TOKEN is not set. Add it to the Lightning Studio environment first."
    )

print(f"Downloading {MODEL_ID} to the persistent Hugging Face cache...")
snapshot_download(repo_id=MODEL_ID, token=HF_TOKEN)

print("Downloading sentence-transformers/all-MiniLM-L6-v2...")
snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2")

print("Model cache ready.")
