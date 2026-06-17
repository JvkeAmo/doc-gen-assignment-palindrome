"""LLM client factory and small completion helpers (OpenAI).

The model and endpoint are env-overridable, so you can swap models (e.g. gpt-4o vs gpt-4o-mini) or
point at an OpenAI-compatible endpoint without code changes:

    LLM_MODEL      default gpt-4o-mini
    LLM_BASE_URL   default https://api.openai.com/v1
    API key        from OPENAI_API_KEY, OPENAI_KEY, or LLM_API_KEY
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def build_client() -> OpenAI:
    """Return an OpenAI client. The API key resolves from OPENAI_API_KEY / OPENAI_KEY / LLM_API_KEY."""
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_KEY")
        or os.environ.get("LLM_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "No API key found. Set OPENAI_API_KEY (or OPENAI_KEY / LLM_API_KEY) in your .env."
        )
    return OpenAI(base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL), api_key=api_key)


def model_name() -> str:
    """The model id to use (env-overridable)."""
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)


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
    """Run one completion and return the text (optionally requesting a JSON-object response)."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {"temperature": temperature}
    if as_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return (response.choices[0].message.content or "").strip()
