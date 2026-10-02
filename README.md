# NCAIR LMS Chatbot 2.0

A text-only multilingual assistant for NCAIR LMS questions. V1 preserves the original keyword router. V2 uses the official `NCAIR1/N-ATLaS` model for language understanding and structured tool selection, then executes the same three deterministic tools against official NCAIR evidence.

**Supported languages:** English · Hausa · Yoruba · Igbo

## Architecture

### V2 — assessed implementation

```mermaid
flowchart LR
    A[User text] --> B[N-ATLaS 4-way language classifier]
    B --> C[N-ATLaS 13-way closed-set route classifier]
    C --> D{Deterministic route mapping}
    D -->|portal| E[get_portal_link]
    D -->|step| F[get_step_guidance]
    D -->|knowledge| G[Multilingual E5 + FAISS]
    G --> H[BGE multilingual reranker]
    H --> I{Calibrated support threshold}
    I -->|supported| J[N-ATLaS grounded answer]
    I -->|unsupported| K[Safe abstention]
    E --> J
    F --> J
    J --> L[Answer in user's language]
```

V2 treats language and routing as closed-set classification instead of free-form tool generation. N-ATLaS scores only allowed choices: four language classes and thirteen mutually exclusive route classes. Route labels map deterministically to portal actions, step numbers, or knowledge search, so invalid JSON and argument drift are eliminated. Knowledge questions keep the original multilingual wording and are embedded with multilingual E5 against a small atomic-fact index derived from the official NCAIR guide. FAISS supplies candidates and a multilingual BGE cross-encoder reranks them. Support is determined by calibrated reranker score/margin thresholds rather than a second generative evidence judge. N-ATLaS is used again only to verbalize verified evidence in the user's language.

### V1 — comparison baseline

```text
User text
  -> keyword/sub-string router
  -> same three tool concepts
  -> FAISS for knowledge questions
  -> Ollama llama3.2:3b for grounded knowledge answers
```

The V1 router is deliberately preserved rather than improved so benchmark comparisons remain meaningful.

## Quick start

Requirements:

- Python 3.11+
- Poppler (`pdftoppm`)
- Tesseract OCR
- enough RAM/VRAM to run the 8B N-ATLaS model for real V2 inference
- Ollama only if you want full V1 answers

```bash
git clone https://github.com/Davemafy/ncair-lms-chatbot2.0.git
cd ncair-lms-chatbot2.0
cp .env.example .env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
make install-dev
```

Without `make`:

```bash
python -m pip install -e ".[runtime,dev]"
```

## Configuration

`.env.example` documents every runtime setting and the application loads `.env` automatically. The defaults select `NCAIR1/N-ATLaS`, `TOP_K=3`, a `0.30` minimum retrieval similarity, and the original Ollama V1 model. `HF_TOKEN` has no default secret value and is optional for public model access.

Environment files are never committed.

## Run V1

Start Ollama first:

```bash
ollama pull llama3.2:3b
ollama serve
```

Then:

```bash
make run-v1
```

Raw command:

```bash
NCAIR_DEFAULT_VERSION=v1 uvicorn ncair_lms.api:app --reload
```

## Run V2

```bash
make run-v2
```

Raw command:

```bash
NCAIR_DEFAULT_VERSION=v2 uvicorn ncair_lms.api:app --reload
```

The first V2 request loads `NCAIR1/N-ATLaS`. The first V2 knowledge request also loads multilingual E5 and the BGE reranker. All model loading is lazy so linting, unit tests, benchmark validation, and API import do not download model weights.

Endpoints:

- `POST /api/v1/chat`
- `POST /api/v2/chat`
- `POST /api/chat` — uses `NCAIR_DEFAULT_VERSION`
- `GET /api/health`

Request:

```json
{"message":"Ina zan shiga LMS?"}
```

Successful response shape:

```json
{
  "answer": "...",
  "language": "hausa",
  "tool": "get_portal_link",
  "sources": ["Verified NCAIR portal registry"]
}
```

## Tests and code quality

```bash
make test
make lint
make worker-test
```

`make check` runs linting/format checks, benchmark validation, offline unit tests, and Worker tests.

Unit tests mock model boundaries. They do not require internet access, N-ATLaS weights, Ollama, Cloudflare, or private credentials.

Real-model smoke tests are opt-in:

```bash
make test-integration
```

## Benchmark

The repository contains a frozen 60-question regression benchmark: 15 each in English, Hausa, Yoruba, and Igbo. After its results have been inspected, it must not be described as a blind or held-out final evaluation set. Use it to detect regressions and measure known failure classes; use a fresh unseen set for final unbiased evaluation.

Category totals:

| Category | Records |
| --- | ---: |
| navigation | 10 |
| step_guidance | 10 |
| documented_policy | 20 |
| trap | 12 |
| undocumented | 8 |

Validate the dataset without model inference:

```bash
make benchmark-validate
```

Run routing comparisons:

```bash
make benchmark-old
make benchmark-new
```

Run the full answer path when both model runtimes are available:

```bash
python -m eval.run_benchmark --version v1 --mode full
python -m eval.run_benchmark --version v2 --mode full
```

Results are written to `eval/results/` and intentionally ignored by Git. The repository does **not** commit fabricated N-ATLaS scores. Generate results on hardware that can actually load the model, then review the output before reporting it.

Metrics include tool accuracy, deterministic-argument accuracy, per-language routing accuracy, language accuracy, trap accuracy, undocumented-question safety, evidence-hit rate, cross-language tool consistency, failure rate, and latency.

## Project structure

```text
.
├── data/                         # official NCAIR sources + atomic retrieval facts
├── eval/                         # held-out benchmark + evaluation code
├── src/
│   └── ncair_lms/
│       ├── api.py                # HTTP boundary
│       ├── config.py             # environment configuration
│       ├── models.py             # shared domain models/enums
│       ├── natlas.py             # N-ATLaS adapter + V2 structured router
│       ├── rag.py                # lazy OCR/embedding/FAISS retrieval
│       ├── service.py            # V1/V2 orchestration
│       ├── tools.py              # canonical deterministic tools
│       └── v1_router.py          # preserved keyword baseline
├── tests/
│   ├── unit/
│   └── integration/
├── web/                          # existing static UI
└── worker/                       # legacy Cloudflare demo runtime
```

## V1 vs V2

| Concern | V1 | V2 |
| --- | --- | --- |
| Router | keyword/sub-string rules | N-ATLaS closed-set route classification |
| Languages | English-oriented baseline | English, Hausa, Yoruba, Igbo |
| Tools | same three concepts | same three canonical tools |
| Knowledge | chunked FAISS official evidence | atomic multilingual FAISS + BGE reranking |
| Knowledge query | original user text | original multilingual user text |
| Answer model | local Ollama | N-ATLaS |
| Evaluation purpose | baseline | assessed implementation |

## Evaluation methodology

The benchmark is data, not prompt material. Runtime prompts contain general contracts and enum constraints, not benchmark questions or phrase-specific routing rules.

For each record the evaluator compares the selected tool, deterministic arguments, detected language, whether retrieved official evidence is sufficient, and failures. Cross-language consistency is measured only for records sharing a `semantic_key`. The frozen 60-case set is a regression suite; final reporting should use a separate unseen holdout after the implementation is frozen.

`routing` mode stops after tool execution. `full` mode additionally generates the answer.

## Actual results

No full N-ATLaS result file is committed because this repository cannot truthfully claim a score without running the 8B model. The evaluation path is reproducible and `eval/results/` is reserved for real run output.

This is preferable to checking in mock or fabricated scores.

## Deployment

The `worker/` directory is the pre-existing Cloudflare Workers AI demo. It uses a Cloudflare-hosted GLM model and is retained so the deployed demo does not disappear. It is **not** the assessed N-ATLaS V2 and must not be reported as such.

The Python V2 is the source of truth for assignment evaluation.

## Limitations

- N-ATLaS is an 8B model; first-run download and inference need appropriate hardware.
- PDF ingestion depends on system-installed Poppler and Tesseract.
- V1 still uses the original English MiniLM chunk index. V2 uses `intfloat/multilingual-e5-base` with the required query/passage prefixes, then `BAAI/bge-reranker-v2-m3` for multilingual reranking.
- V2 route and language decisions are fixed-choice likelihood comparisons rather than generated JSON. Knowledge support is a calibrated reranker decision; `RERANK_MIN_SCORE` and `RERANK_MIN_MARGIN` should be calibrated on development data and frozen before final evaluation.
- The current static Vercel UI points at the legacy Worker deployment; use the Python API endpoints for assessed V1/V2 evaluation.
