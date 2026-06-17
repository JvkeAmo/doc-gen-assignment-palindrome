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

from agent_pipeline.funds import funds_breakdown
from agent_pipeline.models import Account, ClientLedger
from agent_pipeline.runlog import RunRecorder, timed_complete
from agent_pipeline.verify import critique_slot


# --- helpers ---------------------------------------------------------------------------------

def _money(account: Account, ledger: ClientLedger | None = None) -> str:
    if account.value is None:
        # Surface the matching gap as a visible flag, never a blank or a guess.
        if ledger:
            for gap in ledger.gaps:
                if gap.field.startswith(account.account_id):
                    return gap.marker()
        return "[FLAG: balance to be confirmed]"
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
#
# Each renderer returns a dict of named fields. The *wording* of the section lives in the config
# placeholder's "template" (config owns wording); the renderer only supplies values and flags the
# template slots into (Python owns logic). See [[Config Strategy]].

def render_scope(ledger: ClientLedger) -> dict:
    """Field {phrase}: a natural phrase naming the account types the report covers."""
    seen: list[str] = []
    for a in ledger.scoped_accounts():
        label = a.type
        if a.owner.lower() == "joint":
            label = f"jointly-held {a.type}"
        if label not in seen:
            seen.append(label)
    phrase = _natural_join([f"your {s}" for s in seen]) or "your accounts"
    return {"phrase": phrase}


def render_holdings_table(ledger: ClientLedger) -> dict:
    """Field {table}: the holdings table (structural, so it stays code-built).

    Any review flags raised from unrecognised documents are surfaced just below the table, so
    discovered accounts and uncategorised material are visible rather than buried in the ledger.
    """
    rows = ["| Account | Owner | Type | Value |", "|---|---|---|---|"]
    for a in ledger.scoped_accounts():
        rows.append(f"| {a.account_id} | {a.owner} | {a.type} | {_money(a, ledger)} |")
    table = "\n".join(rows)
    review_markers = [gap.marker() for gap in ledger.gaps if gap.review]
    if review_markers:
        table += "\n\n" + "\n".join(review_markers)
    return {"table": table}


def render_cgt_statement(ledger: ClientLedger) -> dict:
    """Field {cgt_flag}: the human-finalise CGT flag(s), or empty when there are none."""
    markers = _gap_markers(ledger, "Tax Implications")
    return {"cgt_flag": ("\n\n" + " ".join(markers)) if markers else ""}


def render_fees(ledger: ClientLedger) -> dict:
    """Fields {initial_charge} (its sentence, when known) and {fee_flags} (the to-confirm flags)."""
    initial = ledger.charges.initial
    initial_charge = f" The initial charge on this recommendation is {initial}." if initial else ""
    markers = _gap_markers(ledger, "Fees & Charges")
    fee_flags = (" " + " ".join(markers)) if markers else ""
    return {"initial_charge": initial_charge, "fee_flags": fee_flags}


def render_next_steps(ledger: ClientLedger) -> dict:
    """Field {items}: loose ends to confirm — accounts the client holds but this report doesn't cover.

    These are surfaced as soft follow-ups (not recommendations and not invented), e.g. an old cash
    account the client mentioned but isn't sure about. Out-of-scope accounts are otherwise dropped.
    """
    items = []
    for a in ledger.out_of_scope_accounts():
        balance = "balance to be confirmed" if a.value is None else f"£{a.value:,.0f}"
        items.append(
            f"- You also hold a {a.type} ({a.account_id}, {balance}) that is not covered by this "
            "report. Please confirm whether it should be included or reviewed separately."
        )
    return {"items": "\n".join(items)}


def render_review(ledger: ClientLedger) -> dict:
    """Fields for the internal reconciliation review sheet.

    This is the audit view of the ledger: it *shows its working* — which source won each conflict and
    why, where every value came from, and every outstanding flag. It is exactly the "which source did
    you trust when sources disagreed" evidence, surfaced as a document rather than buried in the JSON.
    """
    if ledger.conflicts:
        conflicts = "\n".join(
            f"- {c.field}: chose {c.chose} over {c.over} ({c.rule})" for c in ledger.conflicts
        )
    else:
        conflicts = "- None — every value came straight from the system of record."

    source_labels = {
        "db": "system of record (db)",
        "observed": "fresher observed value",
        "unknown": "unrecognised document",
    }
    rows = ["| Account | Value | Source | In scope |", "|---|---|---|---|"]
    for a in ledger.accounts:
        value = "—" if a.value is None else f"£{a.value:,.0f}"
        source = source_labels.get(a.value_source, a.value_source or "—")
        rows.append(f"| {a.account_id} | {value} | {source} | {'yes' if a.in_scope else 'no'} |")
    provenance = "\n".join(rows)

    flags = "\n".join(f"- {gap.marker()}" for gap in ledger.gaps) or "- None."
    return {"client": ledger.client, "conflicts": conflicts, "provenance": provenance, "flags": flags}


RENDERERS = {
    "scope": render_scope,
    "holdings_table": render_holdings_table,
    "cgt_statement": render_cgt_statement,
    "fees": render_fees,
    "next_steps": render_next_steps,
    "review": render_review,
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
    accts = [f"{a.type} ({a.account_id}) currently {_money(a, ledger)}" for a in ledger.scoped_accounts()]
    if accts:
        lines.append("Account values:\n" + "\n".join(f"- {a}" for a in accts))
    if ledger.amounts:
        lines.append("Amounts involved: " + ", ".join(f"£{a:,.0f}" for a in ledger.amounts))
    if ledger.external_funds:
        lines.append("New money available to invest (separate from the existing accounts above):")
        for line in funds_breakdown(ledger.external_funds):
            lines.append(f"- {line}")
    if ledger.disposal:
        # The funds total above is NEW money only. Disinvesting an account also frees its value to
        # reinvest, so the two are separate components — never a single combined figure (which would
        # be an unsourced, computed number the verifier rejects anyway).
        lines.append(
            "Note: the funds to invest come from TWO separate sources — the new money above AND the "
            "proceeds of any account being disinvested (use the account values listed above). State "
            "each relevant figure on its own; do NOT add them into a single total."
        )
    return "\n".join(lines)


_LLM_CONTEXT = {
    "summary": _summary_context,
    "recommendation": _recommendation_context,
}


def generate_with_reflection(
    name: str,
    base_prompt: str,
    ledger: ClientLedger,
    client: OpenAI,
    model: str,
    recorder: RunRecorder | None,
    max_retries: int = 2,
) -> str:
    """Generate a prose slot, critique it against the ledger, and revise on failure.

    Bounded retries; each retry augments the prompt with the specific problems (necessary because at
    temperature 0 an unchanged prompt would just repeat the same output). If it still fails after the
    retries, the section is shipped with a visible review flag rather than looping or hiding the issue.
    """
    prompt = base_prompt
    output = ""
    problems: list[str] = []
    for attempt in range(max_retries + 1):
        label = f"generate:{name}" if attempt == 0 else f"generate:{name}:retry{attempt}"
        output = timed_complete(recorder, label, client, model, prompt)
        problems = critique_slot(name, output, ledger)
        if not problems:
            return output
        prompt = (
            base_prompt
            + "\n\nYour previous attempt had these problems:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\nRewrite the section, fixing them. Use only the figures provided; invent nothing."
        )
    return output + f"\n\n[FLAG: section needs review — {len(problems)} unresolved issue(s)]"


def fill_placeholder(
    name: str,
    spec: dict,
    ledger: ClientLedger,
    client: OpenAI,
    model: str,
    global_instructions: str,
    recorder: RunRecorder | None = None,
) -> str:
    """Resolve one placeholder from its config spec."""
    source = spec.get("source", "llm")
    if source.startswith("render:"):
        # Deterministic slot: the renderer supplies named fields, the config "template" owns the
        # wording they slot into (e.g. "...advice charge.{initial_charge}{fee_flags}").
        fields = RENDERERS[source.split(":", 1)[1]](ledger)
        template = spec.get("template", "{value}")
        return template.format(**fields)
    # llm prose, with a critique → revise loop
    context_fn = _LLM_CONTEXT.get(name)
    context = context_fn(ledger) if context_fn else ""
    base_prompt = f"{global_instructions}\n\n{spec.get('prompt', '')}\n\nFacts:\n{context}"
    return generate_with_reflection(name, base_prompt, ledger, client, model, recorder)
