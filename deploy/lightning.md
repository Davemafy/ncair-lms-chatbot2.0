# Lightning AI deployment

This deployment path runs the canonical Python N-ATLaS V2 implementation inside a Lightning AI Studio.

## Why this path

- no separate backend implementation
- reuses `src/ncair_lms`
- serves the existing web UI and API from one FastAPI origin
- uses a single GPU only when the Studio is switched to GPU
- model files persist in the Studio filesystem across restarts

## 1. Install on the free CPU Studio first

```bash
git clone https://github.com/Davemafy/ncair-lms-chatbot2.0.git
cd ncair-lms-chatbot2.0
git switch feat/lightning-v2

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
```

Configure these as Studio environment variables/secrets:

```text
HF_TOKEN=...
NATLAS_MODEL=NCAIR1/N-ATLaS
NATLAS_DEVICE=auto
NATLAS_QUANTIZATION=4bit
NCAIR_DEFAULT_VERSION=v2
TOP_K=3
MIN_RETRIEVAL_SCORE=0.30
```

The Hugging Face account behind `HF_TOKEN` must have accepted the gated access conditions for `NCAIR1/N-ATLaS`.

## 2. Pre-download model weights on CPU

This avoids spending GPU credits on the initial download.

```bash
python deploy/prefetch_models.py
```

## 3. Switch the Studio to the free T4

On free accounts without a verified payment method, choose the NVIDIA T4 (16 GB).

The deployment sets `NATLAS_QUANTIZATION=4bit` so the official N-ATLaS weights are loaded with bitsandbytes NF4 quantization and fit within the T4's VRAM. This is still N-ATLaS inference, but it is a quantized runtime rather than the original BF16 representation. Keep that distinction explicit in evaluation and documentation.

Then start the app:

```bash
./deploy/start_lightning.sh
```

## 4. Verify locally

```bash
curl http://127.0.0.1:8000/api/health
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/api/v2/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Ina sabo a nan. Ta yaya zan fara?"}'
```

## 5. Expose port 8000

Use Lightning Studio's Ports plugin to expose port `8000` publicly.

The root URL serves the existing frontend. The API remains available under:

- `/api/health`
- `/api/chat`
- `/api/v2/chat`

The frontend uses the same origin, so no separate CORS configuration is required for the Studio-hosted demo.

## Cost control

Keep the Studio on CPU while installing packages and downloading model files. Switch to GPU only immediately before testing or demonstrating N-ATLaS.

Free-tier Studios require periodic restarts. Persistent Studio storage keeps the repository and downloaded model files between restarts.
