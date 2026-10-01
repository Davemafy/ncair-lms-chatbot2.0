# Modal deployment

This deployment serves the canonical N-ATLaS V2 implementation on a Modal L4 GPU.

It deliberately:

- uses the existing `src/ncair_lms` application rather than a second implementation;
- mounts the official `data/` sources and existing `web/` UI;
- caches Hugging Face model files in a persistent Modal Volume;
- limits concurrency to one GPU container;
- scales the GPU container to zero after 30 seconds idle;
- exposes the same `/api/health`, `/api/chat`, `/api/v1/chat`, and `/api/v2/chat` routes.

## Prerequisites

Authenticate Modal and create the Hugging Face secret:

```bash
python -m pip install -U modal
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_your_token
```

The Hugging Face account behind the token must have accepted the gated access conditions for `NCAIR1/N-ATLaS`.

## First deployment

From the repository root:

```bash
modal run deploy/modal_app.py::download_models
modal deploy deploy/modal_app.py
```

The first command downloads N-ATLaS and the embedding model into persistent storage without consuming GPU time.

The second command prints the public Modal URL.

## Smoke test

```bash
curl https://YOUR-MODAL-URL/api/health
```

Then test V2:

```bash
curl -X POST https://YOUR-MODAL-URL/api/v2/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Ina sabo a nan. Ta yaya zan fara?"}'
```

A successful response should report `"language": "hausa"` and must not contain the frontend's `preview mode` label.

## Cost controls

The deployment requests one L4 GPU, permits at most one running GPU container, and scales to zero after 30 seconds idle.

Do not run the complete 60-case benchmark against this public endpoint while the account has only a small credit balance. Use dedicated benchmark compute instead.
