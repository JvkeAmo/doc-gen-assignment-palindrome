"""Run telemetry: capture per-stage timings, every LLM prompt/response, and latency.

Each generation writes a JSON record to ``outputs/runs/<client>_<timestamp>.json`` and appends a row
to ``outputs/runs/index.md`` so runs can be compared at a glance (times, call counts, verification).
This is local dev telemetry for inspecting prompts/outputs/latency — not part of the shipped report.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from agent_pipeline.llm import complete, parse_into
from agent_pipeline.models import ClientLedger


class RunRecorder:
    def __init__(self, client_name: str, model: str) -> None:
        self.client = client_name
        self.model = model
        self.started = datetime.now()
        self._t0 = time.perf_counter()
        # Per-stage wall-clock times (triage/ocr/extract/reconcile/generate/verify). Telemetry only,
        # but it's how we see where the time goes — e.g. that extract dominates, which is what makes
        # parallelising the per-source calls worthwhile. Written to outputs/runs/ + the index table.
        self.stages: dict[str, float] = {}
        self.llm_calls: list[dict] = []
        self._lock = threading.Lock()  # extract now runs its calls in parallel threads

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round(time.perf_counter() - start, 2)

    def record_llm(self, name: str, prompt: str, response: str, seconds: float, system: str | None = None) -> None:
        with self._lock:  # called concurrently during parallel extraction
            self.llm_calls.append(
                {
                    "name": name,
                    "seconds": round(seconds, 2),
                    "system": system,
                    "prompt": prompt,
                    "response": response,
                }
            )

    def finalize(
        self, ledger: ClientLedger | None, report: str, problems: list[str], runs_dir: Path
    ) -> Path:
        total = round(time.perf_counter() - self._t0, 2)
        record = {
            "client": self.client,
            "model": self.model,
            "started": self.started.isoformat(timespec="seconds"),
            "total_seconds": total,
            "stage_seconds": self.stages,
            "llm_seconds_total": round(sum(c["seconds"] for c in self.llm_calls), 2),
            "llm_call_count": len(self.llm_calls),
            "verification": problems if problems else "PASS",
            "llm_calls": self.llm_calls,
            "ledger": ledger.model_dump(mode="json") if ledger is not None else None,
            "report": report,
        }
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.started.strftime("%Y%m%d-%H%M%S")
        path = runs_dir / f"{self.client}_{stamp}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        _append_index(runs_dir, record, stamp)
        return path


def _append_index(runs_dir: Path, record: dict, stamp: str) -> None:
    index = runs_dir / "index.md"
    if not index.exists():
        index.write_text(
            "# Run index\n\n"
            "| When | Client | Model | Total s | Calls | LLM s | Verification |\n"
            "|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    verdict = "PASS" if record["verification"] == "PASS" else f"{len(record['verification'])} issue(s)"
    row = (
        f"| {stamp} | {record['client']} | {record['model']} | {record['total_seconds']} "
        f"| {record['llm_call_count']} | {record['llm_seconds_total']} | {verdict} |\n"
    )
    with index.open("a", encoding="utf-8") as fh:
        fh.write(row)


def timed_complete(
    recorder: RunRecorder | None,
    name: str,
    client: OpenAI,
    model: str,
    prompt: str,
    **kwargs,
) -> str:
    """Run ``complete`` and, if a recorder is given, log the prompt/response and latency."""
    start = time.perf_counter()
    out = complete(client, model, prompt, **kwargs)
    if recorder is not None:
        recorder.record_llm(name, prompt, out, time.perf_counter() - start, kwargs.get("system"))
    return out


def timed_parse(
    recorder: RunRecorder | None,
    name: str,
    client: OpenAI,
    model: str,
    prompt: str,
    response_format: type[BaseModel],
    **kwargs,
):
    """Like ``timed_complete`` but for a structured-output call; returns the parsed object."""
    start = time.perf_counter()
    parsed = parse_into(client, model, prompt, response_format, **kwargs)
    if recorder is not None:
        text = parsed.model_dump_json() if parsed is not None else ""
        recorder.record_llm(name, prompt, text, time.perf_counter() - start, kwargs.get("system"))
    return parsed
