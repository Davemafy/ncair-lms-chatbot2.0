from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .errors import (
    ConfigurationError,
    InvalidModelOutputError,
    InvalidToolArgumentsError,
    ModelUnavailableError,
    RetrievalError,
)
from .service import AssistantService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SETTINGS = Settings.from_env()
WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="NCAIR LMS Chatbot 2.0", version="2.0.0")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@lru_cache(maxsize=2)
def _service(version: str) -> AssistantService:
    return AssistantService(version, settings=SETTINGS)


def _chat(version: str, request: ChatRequest) -> dict:
    try:
        return _service(version).chat(request.message.strip()).as_dict()
    except InvalidModelOutputError as exc:
        raise HTTPException(
            status_code=422,
            detail="The routing model returned an invalid tool decision.",
        ) from exc
    except InvalidToolArgumentsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ModelUnavailableError, RetrievalError, requests.RequestException) as exc:
        raise HTTPException(
            status_code=503,
            detail="The requested assistant backend is temporarily unavailable.",
        ) from exc
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "default_version": SETTINGS.default_version,
        "natlas_model": SETTINGS.natlas_model,
    }


@app.post("/api/v1/chat")
async def chat_v1(request: ChatRequest):
    return _chat("v1", request)


@app.post("/api/v2/chat")
async def chat_v2(request: ChatRequest):
    return _chat("v2", request)


@app.post("/api/chat")
async def chat_default(request: ChatRequest):
    return _chat(SETTINGS.default_version, request)


if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

    @app.get("/styles.css")
    async def stylesheet():
        return FileResponse(WEB_DIR / "styles.css", media_type="text/css")

    @app.get("/app.js")
    async def javascript():
        return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")

    @app.get("/config.js")
    async def frontend_config():
        return FileResponse(WEB_DIR / "config.js", media_type="application/javascript")

    @app.get("/{full_path:path}")
    async def frontend(full_path: str):
        del full_path
        return FileResponse(WEB_DIR / "index.html")
