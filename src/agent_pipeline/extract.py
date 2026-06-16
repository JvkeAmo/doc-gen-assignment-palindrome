"""Extraction: turn the routed source files into structured facts.

By design:
  * the db is parsed **deterministically** (it is already structured, and joint accounts are
    de-duplicated here);
  * each free-text source is read by its own **focused LLM call** — the report request, the meeting
    note (+ internal notes), and any statement image are extracted separately, each anchored by the
    db account digest so entity-matching survives. The partial results are then merged.

Splitting by source keeps each call small and reliable (a small model drops fewer fields on a tight
4-key JSON than a sprawling one), while the shared db anchor preserves the cross-source knowledge.
Extraction does NOT resolve conflicts — it only records what each source says; the db value and any
"live" value both survive, to be reconciled later.
"""

from __future__ import annotations

import json
from datetime import date

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from agent_pipeline.llm import loads_json
from agent_pipeline.models import Account
from agent_pipeline.runlog import RunRecorder, timed_complete
from agent_pipeline.triage import Role


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
    kind: str = "available"  # "available" | "contingent" | "committed"
    note: str | None = None


class DiscoveredAccount(BaseModel):
    """An account an unrecognised document reveals that is NOT in the system-of-record db."""

    account_id: str | None = None
    type: str | None = None
    owner: str | None = None
    value: float | None = None
    as_of: str | None = None
    note: str | None = None


class ExtractedFacts(BaseModel):
    client_label: str | None = None
    risk_profile: str | None = None
    selling: bool | None = None  # report_request "Selling existing investments?"
    initial_charge: str | None = None
    scope_account_ids: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)  # high-level circumstances, no amounts
    investment_amounts: list[float] = Field(default_factory=list)  # headline figures stated in sources
    actions: list[str] = Field(default_factory=list)
    live_values: list[LiveValue] = Field(default_factory=list)
    external_funds: list[ExternalFundLite] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)
    # Populated only by the exploratory pass over unrecognised documents:
    discovered_accounts: list[DiscoveredAccount] = Field(default_factory=list)
    unmapped: list[str] = Field(default_factory=list)  # material found but not categorisable

    @field_validator(
        "scope_account_ids", "objectives", "investment_amounts", "actions", "guidance", "unmapped",
        mode="before",
    )
    @classmethod
    def _coerce_scalar_to_list(cls, v):
        # Models sometimes return a bare string/number where a list is expected.
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    @field_validator("live_values", "external_funds", "discovered_accounts", mode="before")
    @classmethod
    def _coerce_dict_to_list(cls, v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]


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


def _anchor(accounts: list[Account]) -> str:
    return (
        "Accounts in the system-of-record database (use these account_ids):\n"
        + _account_digest(accounts)
    )


def _request_prompt(accounts: list[Account], text: str) -> str:
    return (
        "You are reading a report-requirement summary (an adviser's instruction sheet).\n\n"
        f"{_anchor(accounts)}\n\n=== report_request ===\n{text}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '  "client_label": string naming the client(s), or null;\n'
        '  "risk_profile": the agreed risk profile string, or null;\n'
        '  "selling": true/false/null — does the report involve selling/disposing investments;\n'
        '  "initial_charge": the initial charge string (e.g. "0%"), or null;\n'
        '  "scope_account_ids": list of db account_ids the report covers;\n'
        '  "investment_amounts": monetary amounts written explicitly as figures in THIS document '
        '(e.g. "GBP 20,000" -> 20000). If an amount is only described in words (e.g. "full value of '
        'the GIA"), return []. Do NOT infer numbers from the account database.'
    )


def _meeting_prompt(accounts: list[Account], meeting: str, guidance: str) -> str:
    extra = f"\n=== internal_notes ===\n{guidance}\n" if guidance else ""
    return (
        "You are reading an adviser's free-form meeting note (plus internal data-source notes).\n\n"
        f"{_anchor(accounts)}\n\n=== meeting_notes ===\n{meeting}\n{extra}\n"
        "Return a JSON object with exactly these keys:\n"
        '  "client_label": string naming the client(s), or null;\n'
        '  "objectives": short HIGH-LEVEL circumstance/objective phrases (e.g. "both retired", '
        '"no income required"). Do NOT include amounts;\n'
        '  "actions": short phrases of what the client should do WITH THEIR INVESTMENTS (e.g. '
        '"disinvest the joint GIA in full", "top up both ISAs equally"). Exclude the adviser\'s own '
        'admin steps (e.g. "prepare the report", "confirm the charges");\n'
        '  "live_values": list of {account_id, description, value, as_of, note} for any account '
        "value observed live in the meeting that may differ from the db. Use the meeting date as "
        "as_of. Only when a number is actually given;\n"
        '  "external_funds": list of {label, amount, kind, note} for money not yet an account '
        "(inheritance, business-sale proceeds). kind is one of: \"available\" (an inflow investable "
        'now, e.g. an inheritance or a completion payment received); "contingent" (a future/uncertain '
        'inflow not yet received, e.g. an earnout); "committed" (an amount already earmarked to be '
        'paid out, e.g. a loan repayment);\n'
        '  "guidance": short notes from the internal "## This client" section, or anything to handle '
        "sensitively (e.g. an inheritance following a death)."
    )


def _statement_prompt(accounts: list[Account], text: str) -> str:
    return (
        "You are reading an account statement (possibly OCR'd from an image).\n\n"
        f"{_anchor(accounts)}\n\n=== statement ===\n{text}\n\n"
        "Return a JSON object with exactly this key:\n"
        '  "live_values": list of {account_id, description, value, as_of, note}, one per account row '
        'in the statement. "as_of" is the statement / "valued on" date. Match account_id to the '
        "database where possible; only include rows where a value is given."
    )


def _explore_prompt(accounts: list[Account], text: str) -> str:
    """Prompt for an unrecognised document: discover novel facts, flag what can't be categorised."""
    return (
        "You are reading an UNRECOGNISED client document. We do not know its format, so read it "
        "carefully and extract anything material for an investment advice report.\n\n"
        f"{_anchor(accounts)}\n\n=== unknown_document ===\n{text}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '  "live_values": list of {account_id, description, value, as_of, note} for a value this '
        "document gives for one of the KNOWN accounts above (match account_id where possible);\n"
        '  "discovered_accounts": list of {account_id, type, owner, value, as_of, note} for any '
        "account this document reveals that is NOT in the database above;\n"
        '  "actions": short phrases of what the client should do with their investments;\n'
        '  "guidance": short notes an adviser should handle sensitively;\n'
        '  "unmapped": list of short strings for any material you found but could not place into the '
        "keys above, so a human can review it. Never invent figures."
    )


def _extract_source(
    client: OpenAI, model: str, prompt: str, name: str, recorder: RunRecorder | None
) -> ExtractedFacts:
    """Run one focused per-source extraction call and parse it into ExtractedFacts."""
    raw = timed_complete(recorder, name, client, model, prompt, system=_EXTRACT_SYSTEM, as_json=True)
    try:
        return ExtractedFacts.model_validate(loads_json(raw))
    except (json.JSONDecodeError, ValidationError):
        return ExtractedFacts()  # fail soft: other sources + the db still yield a report


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
    """Run a focused extraction call per source, then merge into one ExtractedFacts."""
    guidance = sources.get(Role.GUIDANCE, "")
    parts: list[ExtractedFacts] = []

    if Role.REPORT_REQUEST in sources:
        prompt = _request_prompt(accounts, sources[Role.REPORT_REQUEST])
        parts.append(_extract_source(client, model, prompt, "extract:report_request", recorder))

    if Role.MEETING_NOTES in sources:
        prompt = _meeting_prompt(accounts, sources[Role.MEETING_NOTES], guidance)
        parts.append(_extract_source(client, model, prompt, "extract:meeting_notes", recorder))

    if Role.IMAGE in sources:
        prompt = _statement_prompt(accounts, sources[Role.IMAGE])
        parts.append(_extract_source(client, model, prompt, "extract:statement", recorder))

    # Exploratory pass over unrecognised documents — fires only when one is present, so the normal
    # case pays nothing. It can discover accounts not in the db and flag material it can't categorise.
    if Role.UNKNOWN in sources:
        prompt = _explore_prompt(accounts, sources[Role.UNKNOWN])
        parts.append(_extract_source(client, model, prompt, "extract:unknown", recorder))

    return merge_facts(parts)
