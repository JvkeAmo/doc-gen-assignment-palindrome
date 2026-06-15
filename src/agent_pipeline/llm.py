"""LLM client factory and small completion helpers.

Defaults to a **local Ollama** server, which exposes an OpenAI-compatible API, so that
development and testing do not consume hosted API credits. Point the ``LLM_*`` environment
variables elsewhere (e.g. OpenAI) to switch provider without code changes.

    LLM_BASE_URL   default http://localhost:11434/v1   (local Ollama)
    LLM_API_KEY    default "ollama"                     (Ollama ignores the value)
    LLM_MODEL      default qwen3:8b
"""

from __future__ import annotations

import os
import re

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:8b"

# qwen3 and similar "thinking" models emit a <think>...</think> preamble we don't want in output.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_client() -> OpenAI:
    """Return an OpenAI-compatible client, pointed at Ollama by default."""
    return OpenAI(
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("LLM_API_KEY", DEFAULT_API_KEY),
    )


def model_name() -> str:
    """The model id to use (env-overridable)."""
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)


def strip_thinking(text: str | None) -> str:
    """Remove any <think>...</think> block and surrounding whitespace."""
    if not text:
        return ""
    return _THINK_RE.sub("", text).strip()
