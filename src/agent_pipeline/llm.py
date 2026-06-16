"""LLM client factory and small completion helpers.

Defaults to a **local Ollama** server, which exposes an OpenAI-compatible API, so that
development and testing do not consume hosted API credits. Point the ``LLM_*`` environment
variables elsewhere (e.g. OpenAI) to switch provider without code changes.

    LLM_BASE_URL   default http://localhost:11434/v1   (local Ollama)
    LLM_API_KEY    default "ollama"                     (Ollama ignores the value)
    LLM_MODEL      default qwen3:8b
"""

from __future__ import annotations

import json
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


def loads_json(raw: str) -> dict:
    """Tolerant JSON parse of a model response: strip code fences and isolate the outermost object."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def complete(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    as_json: bool = False,
    temperature: float = 0.0,
) -> str:
    """Run one completion and return cleaned text.

    Disables qwen3's reasoning preamble (via ``/no_think``) for speed and clean output, and
    optionally requests a JSON object response.
    """
    user = prompt
    if model.startswith("qwen3"):
        user = f"{user}\n\n/no_think"

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    kwargs: dict = {"temperature": temperature}
    if as_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return strip_thinking(response.choices[0].message.content)
