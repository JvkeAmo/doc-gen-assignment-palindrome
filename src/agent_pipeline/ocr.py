"""Read text out of an image source (e.g. a statement screenshot).

Uses a vision-capable model (default ``gpt-4o-mini``, env-overridable via ``OCR_MODEL``). Fails soft:
if OCR is unavailable, the caller still gets a report from the db and prose sources alone.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

from agent_pipeline.llm import build_client
from agent_pipeline.runlog import RunRecorder

_OCR_PROMPT = (
    "Extract all text and tables from this account statement image as plain markdown. "
    "Include every account name, type, value, and date exactly as shown."
)


def read_image_text(path: Path, recorder: RunRecorder | None = None) -> str:
    """Return OCR'd text for an image, or a short placeholder if OCR fails."""
    model = os.environ.get("OCR_MODEL", "gpt-4o-mini")
    start = time.perf_counter()
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
        text = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 - OCR is best-effort
        text = f"[image OCR unavailable for {path.name}: {exc}]"
    if recorder is not None:
        recorder.record_llm(f"ocr:{path.name}", _OCR_PROMPT, text, time.perf_counter() - start, model)
    return text
