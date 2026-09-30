"""Compatibility module for the original ``uvicorn src.api:app`` command."""

from ncair_lms.api import app

__all__ = ["app"]
