"""Extraction: turn the routed source files into structured facts.

Two paths, by design:
  * the db is parsed **deterministically** (it is already structured, and joint accounts are
    de-duplicated here);
  * the free-text sources (meeting notes, report request, per-client guidance, and any OCR'd
    image text) are read by **one LLM call** that returns a typed ``ExtractedFacts`` JSON.

Extraction does NOT resolve conflicts — it only records what each source says. The db value and
any "live" value from the notes both survive, to be reconciled later.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from agent_pipeline.llm import complete
from agent_pipeline.models import Account
from agent_pipeline.triage import Role
from document_formatter.loading import read_file


# --- LLM extraction output schema ------------------------------------------------------------

class LiveValue(BaseModel):
    """An account value observed in the notes/image (often fresher than the db snapshot)."""

    account_id: str | None = None  # db id when identifiable
    description: str | None = None  # e.g. "joint GIA" when no id was given
    value: float | None = None
    as_of: str | None = None  # ISO date the figure was observed (usually the meeting date)
    note: str | None = None


class ExternalFundLite(BaseModel):
    label: str
    amount: float | None = None
    available: bool = True
    note: str | None = None


class ExtractedFacts(BaseModel):
    client_label: str | None = None
    risk_profile: str | None = None
    selling: bool | None = None  # report_request "Selling existing investments?"
    initial_charge: str | None = None
    scope_account_ids: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)  # high-level circumstances, no amounts
    actions: list[str] = Field(default_factory=list)
    live_values: list[LiveValue] = Field(default_factory=list)
    external_funds: list[ExternalFundLite] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)


# --- deterministic db parse ------------------------------------------------------------------

def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def parse_db(text: str) -> tuple[list[Account], date | None]:
    """Parse the custody db into de-duplicated accounts (+ the snapshot date).

    A joint account is listed under each holder; we keep it once.
    """
    data = json.loads(text)
    snapshot = _parse_date(data.get("snapshot_date"))
    by_id: dict[str, Account] = {}
    for holder in data.get("holders", {}).values():
        for acc in holder.get("accounts", []):
            aid = acc.get("account_id")
            if not aid or aid in by_id:
                continue  # de-dupe joint accounts
            by_id[aid] = Account(
                account_id=aid,
                owner=acc.get("owner", ""),
                type=acc.get("type", ""),
                platform=acc.get("platform"),
                value=acc.get("value"),
                currency=acc.get("currency", "GBP"),
                valuation_date=_parse_date(acc.get("valuation_date")),
                status=acc.get("status", "open"),
                value_source="db",
            )
    return list(by_id.values()), snapshot


# --- LLM extraction of the prose sources -----------------------------------------------------

_EXTRACT_SYSTEM = (
    "You extract structured facts from UK financial-adviser documents. "
    "Return ONLY a JSON object. Never invent figures: if a value is not stated, use null. "
    "Do not include capital gains tax amounts or fee rates (a person finalises those)."
)


def _account_digest(accounts: list[Account]) -> str:
    lines = []
    for a in accounts:
        val = "blank" if a.value is None else f"{a.value:.0f} {a.currency}"
        lines.append(
            f"- {a.account_id}: {a.type}, owner {a.owner}, platform {a.platform}, "
            f"value {val} as of {a.valuation_date}, status {a.status}"
        )
    return "\n".join(lines)


def _build_extract_prompt(accounts: list[Account], prose_sources: dict[str, str]) -> str:
    parts = [
        "Accounts known from the system-of-record database (use these account_ids):",
        _account_digest(accounts),
        "",
        "Source documents follow. Read them and extract the facts.",
    ]
    for name, text in prose_sources.items():
        parts.append(f"\n=== {name} ===\n{text}")
    parts.append(
        "\nReturn a JSON object with these keys:\n"
        '  "client_label": string naming the client(s), e.g. "David & Susan Clarke";\n'
        '  "risk_profile": the agreed risk profile string, or null;\n'
        '  "selling": true/false/null — does the report involve selling/disposing investments;\n'
        '  "initial_charge": the initial charge string (e.g. "0%"), or null;\n'
        '  "scope_account_ids": list of db account_ids the report covers;\n'
        '  "objectives": short HIGH-LEVEL phrases about the client\'s circumstances and objectives '
        '(e.g. "both retired", "no income required from the portfolio", "long-term growth"). Do '
        "NOT include any amounts here;\n"
        '  "actions": short phrases describing what the client should do (e.g. "disinvest the '
        'joint GIA in full", "top up both ISAs equally");\n'
        '  "live_values": list of {account_id, description, value, as_of, note} for any account '
        "value observed during the meeting or in a statement that may differ from the db. Use the "
        "meeting date as as_of. Only include when a number is actually given;\n"
        '  "external_funds": list of {label, amount, available, note} for money that is not yet an '
        'account (an inheritance, business-sale proceeds). Set available=false for money not yet '
        "received (e.g. a contingent earnout) or already committed elsewhere;\n"
        '  "guidance": list of short notes from the internal "## This client" section or anything '
        "an adviser should handle sensitively (e.g. an inheritance following a death)."
    )
    return "\n".join(parts)


def _loads_json(raw: str) -> dict:
    """Tolerant JSON parse: strip code fences and isolate the outermost object."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def extract_facts(
    client: OpenAI,
    model: str,
    accounts: list[Account],
    prose_sources: dict[str, str],
) -> ExtractedFacts:
    """One LLM call over the filtered prose sources → typed facts."""
    prompt = _build_extract_prompt(accounts, prose_sources)
    raw = complete(client, model, prompt, system=_EXTRACT_SYSTEM, as_json=True)
    try:
        return ExtractedFacts.model_validate(_loads_json(raw))
    except (json.JSONDecodeError, ValidationError):
        # Fail soft: an empty facts object still yields a report from the db alone.
        return ExtractedFacts()


def read_sources(grouped: dict[Role, list[Path]], image_text: dict[str, str] | None = None) -> tuple[str, dict[str, str]]:
    """Return (db_text, prose_sources) from triaged files. ``image_text`` maps filename→OCR text."""
    db_text = ""
    prose: dict[str, str] = {}
    for role, paths in grouped.items():
        for path in paths:
            if role is Role.DB:
                db_text = read_file(path)
            elif role in {Role.MEETING_NOTES, Role.REPORT_REQUEST, Role.GUIDANCE}:
                prose[path.name] = read_file(path)
            elif role is Role.IMAGE and image_text and path.name in image_text:
                prose[path.name] = image_text[path.name]
    return db_text, prose
