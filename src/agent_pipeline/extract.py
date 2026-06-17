"""Extraction: turn the routed source files into structured facts.

By design:
  * the db is parsed **deterministically** (it is already structured, and joint accounts are
    de-duplicated here);
  * each free-text source is read by its own **focused LLM call** — the report request, the meeting
    note (+ internal notes), and any statement image are extracted separately, each anchored by the
    db account digest so entity-matching survives. The partial results are then merged.

Each prose call uses OpenAI **structured outputs** against its OWN per-source schema (see the schema
classes below), so the model can only return fields relevant to that source — no field bleed across
calls, and the shape is schema-guaranteed (no tolerant parsing or coercion needed). The shared db
anchor preserves cross-source knowledge. Extraction does NOT resolve conflicts — it only records what
each source says; the db value and any "live" value both survive, to be reconciled later.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from agent_pipeline.models import Account
from agent_pipeline.runlog import RunRecorder, timed_parse
from agent_pipeline.triage import Role


# --- Extraction schemas ----------------------------------------------------------------------
#
# Each prose source is read with OpenAI structured outputs (llm.parse_into) against its OWN schema,
# so a call can only return fields relevant to that source — no field bleed across calls, and no
# tolerant parsing/coercion needed (the shape is schema-guaranteed). The per-field descriptions
# carry the extraction RULES: the schema is the contract. Partials are merged into ExtractedFacts.

class LiveValue(BaseModel):
    """An account value observed in the notes/image (often fresher than the db snapshot)."""

    account_id: str | None = Field(None, description="db account_id when identifiable")
    description: str | None = Field(None, description='e.g. "joint GIA" when no id was given')
    value: float | None = None
    as_of: str | None = Field(None, description="ISO date the figure was observed (usually the meeting date)")
    note: str | None = None


class ExternalFundLite(BaseModel):
    """Money not yet an account (an inheritance, business-sale proceeds)."""

    label: str
    amount: float | None = None
    kind: Literal["available", "contingent", "committed"] = Field(
        "available",
        description=(
            "available = an inflow investable now (inheritance, completion payment received); "
            "contingent = a future/uncertain inflow not yet received (an earnout); "
            "committed = an amount already earmarked to be paid out (a loan repayment)"
        ),
    )
    note: str | None = None


class DiscoveredAccount(BaseModel):
    """An account an unrecognised document reveals that is NOT in the system-of-record db."""

    account_id: str | None = None
    type: str | None = None
    owner: str | None = None
    value: float | None = None
    as_of: str | None = None
    note: str | None = None


_LIVE_VALUE_RULE = (
    "account values observed that may differ from the db. Record ONLY a balance explicitly stated as "
    "the current figure — never compute or project one (e.g. a balance after a planned transfer or "
    "sale). Match account_id to the db where possible."
)


class RequestFacts(BaseModel):
    """Facts from the report-requirement summary (the adviser's instruction sheet)."""

    client_label: str | None = Field(None, description="the client(s) named, e.g. 'David & Susan Clarke'")
    risk_profile: str | None = Field(None, description="the agreed risk profile string")
    selling: bool | None = Field(None, description="does the report involve selling/disposing investments?")
    initial_charge: str | None = Field(None, description='the initial charge string, e.g. "0%"')
    scope_account_ids: list[str] = Field(
        default_factory=list,
        description=(
            "db account_ids the report covers, per the 'Accounts covered' line. Include ONLY accounts "
            "named there; exclude any account the client merely holds that is not named as covered "
            "(e.g. an unrelated or source-of-funds cash account)."
        ),
    )
    investment_amounts: list[float] = Field(
        default_factory=list,
        description=(
            "monetary amounts written explicitly as figures in THIS document (e.g. 'GBP 20,000' -> "
            "20000). If an amount is only described in words ('full value of the GIA'), leave empty. "
            "Never infer numbers from the account database."
        ),
    )


class MeetingFacts(BaseModel):
    """Facts from the adviser's meeting note (+ internal data-source notes)."""

    client_label: str | None = None
    objectives: list[str] = Field(
        default_factory=list,
        description=(
            "short HIGH-LEVEL circumstance/objective phrases (e.g. 'both retired', 'no income "
            "required'). No amounts. Exclude future aspirations the client is NOT acting on in this "
            "report (gifts, donations, or purchases mentioned only in passing)."
        ),
    )
    actions: list[str] = Field(
        default_factory=list,
        description=(
            "short phrases of what the client should do WITH THEIR INVESTMENTS (e.g. 'disinvest the "
            "joint GIA in full'). Exclude the adviser's own admin steps (e.g. 'prepare the report')."
        ),
    )
    live_values: list[LiveValue] = Field(default_factory=list, description=_LIVE_VALUE_RULE)
    external_funds: list[ExternalFundLite] = Field(default_factory=list)
    guidance: list[str] = Field(
        default_factory=list,
        description=(
            "genuinely sensitive circumstances to handle tactfully (e.g. an inheritance following a "
            "death). Do NOT include future aspirations the client is not acting on (gifts, donations, "
            "purchases) — those are distractions, not advice guidance."
        ),
    )


class StatementFacts(BaseModel):
    """Facts from an account statement (often OCR'd from an image)."""

    live_values: list[LiveValue] = Field(
        default_factory=list,
        description="one per account row in the statement; as_of is the 'valued on' date. " + _LIVE_VALUE_RULE,
    )


class ExploreFacts(BaseModel):
    """Facts from an UNRECOGNISED document: discover what's novel, flag what can't be placed."""

    live_values: list[LiveValue] = Field(
        default_factory=list, description="values this document gives for KNOWN db accounts"
    )
    discovered_accounts: list[DiscoveredAccount] = Field(
        default_factory=list, description="accounts this document reveals that are NOT in the db"
    )
    actions: list[str] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)
    unmapped: list[str] = Field(
        default_factory=list, description="material found but not categorisable, for a human to review"
    )


class ExtractedFacts(BaseModel):
    """The merged result of all per-source extractions (every field union'd together)."""

    client_label: str | None = None
    risk_profile: str | None = None
    selling: bool | None = None
    initial_charge: str | None = None
    scope_account_ids: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    investment_amounts: list[float] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    live_values: list[LiveValue] = Field(default_factory=list)
    external_funds: list[ExternalFundLite] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)
    discovered_accounts: list[DiscoveredAccount] = Field(default_factory=list)
    unmapped: list[str] = Field(default_factory=list)


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
    "You extract structured facts from UK financial-adviser documents into the given schema. "
    "Never invent figures: if a value is not stated, leave it null/empty. Do not include capital "
    "gains tax amounts or fee rates (a person finalises those)."
)


def _anchor(accounts: list[Account]) -> str:
    """The db account digest each per-source prompt is anchored to, to preserve entity-matching."""
    lines = ["Accounts in the system-of-record database (use these account_ids):"]
    for a in accounts:
        val = "blank" if a.value is None else f"{a.value:.0f} {a.currency}"
        lines.append(
            f"- {a.account_id}: {a.type}, owner {a.owner}, platform {a.platform}, "
            f"value {val} as of {a.valuation_date}, status {a.status}"
        )
    return "\n".join(lines)


def _request_prompt(accounts: list[Account], text: str) -> str:
    return (
        "Read this report-requirement summary (an adviser's instruction sheet) and extract its facts.\n\n"
        f"{_anchor(accounts)}\n\n=== report_request ===\n{text}"
    )


def _meeting_prompt(accounts: list[Account], meeting: str, guidance: str) -> str:
    extra = f"\n=== internal_notes ===\n{guidance}\n" if guidance else ""
    return (
        "Read this adviser's free-form meeting note (plus internal data-source notes) and extract its "
        f"facts.\n\n{_anchor(accounts)}\n\n=== meeting_notes ===\n{meeting}\n{extra}"
    )


def _statement_prompt(accounts: list[Account], text: str) -> str:
    return (
        "Read this account statement (possibly OCR'd from an image) and extract a live value per "
        f"account row.\n\n{_anchor(accounts)}\n\n=== statement ===\n{text}"
    )


def _explore_prompt(accounts: list[Account], text: str) -> str:
    return (
        "Read this UNRECOGNISED client document. We do not know its format, so read it carefully and "
        "extract anything material for an investment advice report, flagging what you cannot place.\n\n"
        f"{_anchor(accounts)}\n\n=== unknown_document ===\n{text}"
    )


def _extract_source(
    client: OpenAI,
    model: str,
    prompt: str,
    schema: type[BaseModel],
    name: str,
    recorder: RunRecorder | None,
) -> ExtractedFacts:
    """Run one focused per-source extraction (structured output) and lift it into ExtractedFacts."""
    try:
        facts = timed_parse(recorder, name, client, model, prompt, schema, system=_EXTRACT_SYSTEM)
        return ExtractedFacts(**facts.model_dump()) if facts is not None else ExtractedFacts()
    except Exception:  # noqa: BLE001 - fail soft: other sources + the db still yield a report
        return ExtractedFacts()


def merge_facts(parts: list[ExtractedFacts]) -> ExtractedFacts:
    """Combine per-source partials: first non-null wins for scalars, union for lists."""
    out = ExtractedFacts()
    for p in parts:
        out.client_label = out.client_label or p.client_label
        out.risk_profile = out.risk_profile or p.risk_profile
        if out.selling is None:
            out.selling = p.selling
        out.initial_charge = out.initial_charge or p.initial_charge
        out.scope_account_ids += [x for x in p.scope_account_ids if x not in out.scope_account_ids]
        out.objectives += p.objectives
        out.investment_amounts += [x for x in p.investment_amounts if x not in out.investment_amounts]
        out.actions += p.actions
        out.live_values += p.live_values
        out.external_funds += p.external_funds
        out.guidance += p.guidance
        out.discovered_accounts += p.discovered_accounts
        out.unmapped += p.unmapped
    return out


def extract_facts(
    client: OpenAI,
    model: str,
    accounts: list[Account],
    sources: dict[Role, str],
    recorder: RunRecorder | None = None,
) -> ExtractedFacts:
    """Run a focused extraction call per source, concurrently, then merge into one ExtractedFacts.

    The per-source calls are independent and I/O-bound on the model server, so they run in parallel
    threads — this turns the extract stage from sum-of-calls into about the slowest single call.
    Results are gathered back in a FIXED source order so the first-non-null merge stays deterministic
    no matter which call returns first.
    """
    guidance = sources.get(Role.GUIDANCE, "")

    # (telemetry_name, prompt, schema) in priority order — this order decides who wins a first-non-null
    # scalar in merge_facts, so it must not depend on completion order.
    tasks: list[tuple[str, str, type[BaseModel]]] = []
    if Role.REPORT_REQUEST in sources:
        tasks.append(("extract:report_request", _request_prompt(accounts, sources[Role.REPORT_REQUEST]), RequestFacts))
    if Role.MEETING_NOTES in sources:
        tasks.append(("extract:meeting_notes", _meeting_prompt(accounts, sources[Role.MEETING_NOTES], guidance), MeetingFacts))
    if Role.IMAGE in sources:
        tasks.append(("extract:statement", _statement_prompt(accounts, sources[Role.IMAGE]), StatementFacts))
    # Exploratory pass over unrecognised documents — only present when an unknown file exists, so the
    # normal case pays nothing. It can discover accounts not in the db and flag uncategorised material.
    if Role.UNKNOWN in sources:
        tasks.append(("extract:unknown", _explore_prompt(accounts, sources[Role.UNKNOWN]), ExploreFacts))
    if not tasks:
        return ExtractedFacts()

    # Run the independent calls concurrently. pool.map keeps results in task order, so the
    # first-non-null merge stays deterministic regardless of which call returns first.
    def run(task: tuple[str, str, type[BaseModel]]) -> ExtractedFacts:
        name, prompt, schema = task
        return _extract_source(client, model, prompt, schema, name, recorder)

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        parts = list(pool.map(run, tasks))
    return merge_facts(parts)
