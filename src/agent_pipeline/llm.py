"""LLM client factory and small completion helpers (OpenAI).

Two models, both env-overridable: a stronger one for extraction (precision matters — getting scope
and figures right) and a cheaper one for generation (prose):

    EXTRACT_MODEL  default gpt-4o        (extraction)
    LLM_MODEL      default gpt-4o-mini   (generation)
    LLM_BASE_URL   default https://api.openai.com/v1
    API key        from OPENAI_API_KEY, OPENAI_KEY, or LLM_API_KEY
"""

from __future__ import annotations

import json
import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EXTRACT_MODEL = "gpt-4o"

_Model = TypeVar("_Model", bound=BaseModel)


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
    """The generation model id (env-overridable)."""
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)


def extract_model_name() -> str:
    """The extraction model id — a stronger model than generation, since precision matters here."""
    return os.environ.get("EXTRACT_MODEL", DEFAULT_EXTRACT_MODEL)


def parse_into(
    client: OpenAI,
    model: str,
    prompt: str,
    response_format: type[_Model],
    *,
    system: str | None = None,
    temperature: float = 0.0,
) -> _Model | None:
    """Structured-output completion: return the response parsed into ``response_format``.

    OpenAI guarantees the JSON matches the model's schema, so there is no tolerant parsing or
    scalar/dict coercion to do — a list field is always a list, etc. Returns ``None`` on a refusal.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.parse(
        model=model, messages=messages, response_format=response_format, temperature=temperature
    )
    return completion.choices[0].message.parsed


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
