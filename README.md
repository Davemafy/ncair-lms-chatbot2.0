# NCAIR LMS Assistant 2.0

Railway-ready packaging of the existing NCAIR LMS Assistant v1 backend with a custom web presentation layer.

## Architecture

- FastAPI serves both the web interface and `/api/chat`.
- The original keyword router selects link, onboarding-step or knowledge-base behaviour.
- FAISS and `sentence-transformers/all-MiniLM-L6-v2` provide semantic retrieval.
- Tesseract and Poppler ingest the supplied PDF manual.
- Ollama `llama3.2:3b` uses `/api/generate` to answer from retrieved context.
- The existing `get_portal_link`, `get_step_guidance`, and `search_ncair_knowledge_base` behaviours are preserved.

## Run locally

```bash
ollama pull llama3.2:3b
ollama serve
pip install -r requirements.txt
uvicorn src.api:app --reload
```

Open `http://localhost:8000`.

## Railway

Deploy this repository as the API/web service and set:

```text
OLLAMA_BASE_URL=https://your-ollama-service
OLLAMA_MODEL=llama3.2:3b
```

The Ollama URL must be reachable from the Railway service. Keep it private inside the same Railway project when possible.
