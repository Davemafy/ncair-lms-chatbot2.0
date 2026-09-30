from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return value


def _unit_interval_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if not 0 <= value <= 1:
        raise ConfigurationError(f"{name} must be between 0 and 1.")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    natlas_model: str
    natlas_device: str
    natlas_max_new_tokens: int
    hf_token: str | None
    top_k: int
    min_retrieval_score: float
    ollama_base_url: str
    ollama_model: str
    default_version: str

    @classmethod
    def from_env(cls) -> "Settings":
        repository_root = Path(__file__).resolve().parents[2]
        default_version = os.getenv("NCAIR_DEFAULT_VERSION", "v2").strip().lower()
        if default_version not in {"v1", "v2"}:
            raise ConfigurationError("NCAIR_DEFAULT_VERSION must be either 'v1' or 'v2'.")

        return cls(
            data_dir=Path(os.getenv("NCAIR_DATA_DIR", repository_root / "data")).resolve(),
            natlas_model=os.getenv("NATLAS_MODEL", "NCAIR1/N-ATLaS").strip(),
            natlas_device=os.getenv("NATLAS_DEVICE", "auto").strip(),
            natlas_max_new_tokens=_positive_int("NATLAS_MAX_NEW_TOKENS", 320),
            hf_token=os.getenv("HF_TOKEN") or None,
            top_k=_positive_int("TOP_K", 3),
            min_retrieval_score=_unit_interval_float("MIN_RETRIEVAL_SCORE", 0.30),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip(),
            default_version=default_version,
        )
