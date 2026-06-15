"""Generation off the ledger.

Each section placeholder is filled from a *slice* of the reconciled ``ClientLedger`` rather than
from raw files, so decoys and unresolved conflicts can never reach the report. Two kinds of slot:

  * ``render:<name>``  — a deterministic renderer (tables, fees, scope, the CGT statement). No LLM,
    so figures are exact and the human-finalise gaps are always shown as flags.
  * ``llm``            — prose (summary, recommendation) written by the model from a tight,
    fact-only context. Verbatim lines (FCA, risk warning) are static template text, never generated.
"""

from __future__ import annotations

from openai import OpenAI

from agent_pipeline.llm import complete
from agent_pipeline.models import Account, ClientLedger, Gap


# --- helpers ---------------------------------------------------------------------------------

def _money(account: Account) -> str:
    if account.value is None:
        return "to be confirmed"
    text = f"£{account.value:,.0f}"
    return f"{text} (approx.)" if account.approximate else text


def _gap_markers(ledger: ClientLedger, section: str) -> list[str]:
    return [g.marker() for g in ledger.gaps if g.section == section]


def _natural_join(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# --- deterministic renderers -----------------------------------------------------------------

def render_scope(ledger: ClientLedger) -> str:
    """A natural phrase naming the account types the report covers."""
    seen: list[str] = []
    for a in ledger.scoped_accounts():
        label = a.type
        if a.owner.lower() == "joint":
            label = f"jointly-held {a.type}"
        if label not in seen:
            seen.append(label)
    phrase = _natural_join([f"your {s}" for s in seen])
    return phrase or "your accounts"


def render_holdings_table(ledger: ClientLedger) -> str:
    rows = ["| Account | Owner | Type | Value |", "|---|---|---|---|"]
    for a in ledger.scoped_accounts():
        rows.append(f"| {a.account_id} | {a.owner} | {a.type} | {_money(a)} |")
    return "\n".join(rows)


def render_cgt_statement(ledger: ClientLedger) -> str:
    base = (
        "The recommended disposal may give rise to a capital gains tax liability, which would be "
        "assessed against your annual exempt amount."
    )
    markers = _gap_markers(ledger, "Tax Implications")
    return base + ("\n\n" + " ".join(markers) if markers else "")


def render_fees(ledger: ClientLedger) -> str:
    parts = [
        "The ongoing charges that apply are the platform charge levied by the platform and our "
        "ongoing advice charge."
    ]
    if ledger.charges.initial:
        parts.append(f"The initial charge on this recommendation is {ledger.charges.initial}.")
    markers = _gap_markers(ledger, "Fees & Charges")
    if markers:
        parts.append(" ".join(markers))
    return " ".join(parts)


RENDERERS = {
    "scope": render_scope,
    "holdings_table": render_holdings_table,
    "cgt_statement": render_cgt_statement,
    "fees": render_fees,
}


# --- LLM prose ------------------------------------------------------------------------------

def _summary_context(ledger: ClientLedger) -> str:
    lines = [f"Client: {ledger.client}"]
    if ledger.risk_profile:
        lines.append(f"Agreed risk profile: {ledger.risk_profile}")
    if ledger.objectives_text():
        lines.append("Objectives / circumstances: " + ledger.objectives_text())
    if ledger.guidance:
        lines.append("Sensitive guidance to reflect tactfully: " + "; ".join(ledger.guidance))
    types = _natural_join([a.type for a in ledger.scoped_accounts()])
    if types:
        lines.append(f"Accounts covered (types only): {types}")
    return "\n".join(lines)


def _recommendation_context(ledger: ClientLedger) -> str:
    lines = []
    if ledger.actions:
        lines.append("Recommended actions:")
        for a in ledger.actions:
            lines.append(f"- {a.text}")
    accts = [f"{a.type} ({a.account_id}) currently {_money(a)}" for a in ledger.scoped_accounts()]
    if accts:
        lines.append("Account values:\n" + "\n".join(f"- {a}" for a in accts))
    if ledger.external_funds:
        lines.append("Other funds:")
        for f in ledger.external_funds:
            amt = f"£{f.amount:,.0f}" if f.amount is not None else "amount to confirm"
            avail = "available now" if f.available else "not yet available / committed"
            lines.append(f"- {f.label}: {amt} ({avail}){' — ' + f.note if f.note else ''}")
    return "\n".join(lines)


_LLM_CONTEXT = {
    "summary": _summary_context,
    "recommendation": _recommendation_context,
}


def fill_placeholder(
    name: str, spec: dict, ledger: ClientLedger, client: OpenAI, model: str, global_instructions: str
) -> str:
    """Resolve one placeholder from its config spec."""
    source = spec.get("source", "llm")
    if source.startswith("render:"):
        return RENDERERS[source.split(":", 1)[1]](ledger)
    # llm prose
    context_fn = _LLM_CONTEXT.get(name)
    context = context_fn(ledger) if context_fn else ""
    prompt = f"{global_instructions}\n\n{spec.get('prompt', '')}\n\nFacts:\n{context}"
    return complete(client, model, prompt)
