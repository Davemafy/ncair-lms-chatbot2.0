from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .v1_backend import OLLAMA_BASE_URL, OLLAMA_MODEL, agentic_rag_orchestrator


app = FastAPI(title="NCAIR LMS Assistant API", version="2.0.0")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@app.get("/api/health")
async def health():
    return {"status": "ok", "backend": "v1", "model": OLLAMA_MODEL}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        return agentic_rag_orchestrator(request.message.strip())
    except (requests.RequestException, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="The local language model is not ready yet. Please try again shortly.",
        ) from error


app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/{full_path:path}")
async def frontend(full_path: str):
    return FileResponse(WEB_DIR / "index.html")
