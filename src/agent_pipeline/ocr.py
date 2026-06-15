"""Read text out of an image source (e.g. a statement screenshot).

Uses a local vision/OCR model via the same OpenAI-compatible endpoint as the rest of the
pipeline (default ``deepseek-ocr`` on Ollama). Fails soft: if OCR is unavailable, the caller
still gets a report from the db and prose sources alone.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from agent_pipeline.llm import build_client, strip_thinking

_OCR_PROMPT = (
    "Extract all text and tables from this account statement image as plain markdown. "
    "Include every account name, type, value, and date exactly as shown."
)


def read_image_text(path: Path) -> str:
    """Return OCR'd text for an image, or a short placeholder if OCR fails."""
    model = os.environ.get("OCR_MODEL", "deepseek-ocr:latest")
    try:
        b64 = base64.b64encode(path.read_bytes()).decode()
        response = build_client().chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            temperature=0,
        )
        return strip_thinking(response.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001 - OCR is best-effort
        return f"[image OCR unavailable for {path.name}: {exc}]"
