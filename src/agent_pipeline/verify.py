"""Verification: deterministic checks of a generated report against its ledger.

These are the rules we decided define a "correct" report. They run as an eval (offline, over the
example clients) and can also gate a report before it is sent. Each check returns a list of
problems; an empty list means the report passed.

Code can verify far more than it might seem: verbatim text, section presence vs. the disposal
flag, that no decoy/stale figures leaked, that human-finalise figures appear as flags rather than
invented numbers, and that the holdings table matches the ledger.
"""

from __future__ import annotations

import re

from agent_pipeline.models import ClientLedger

FCA_LINE = "This firm is authorised and regulated by the Financial Conduct Authority."
RISK_WARNING = (
    "The value of investments can fall as well as rise and you may get back less than you invest. "
    "Past performance is not a guide to future returns."
)

def money_tokens(text: str) -> set[str]:
    """Monetary amounts in the text, normalised to digits-only (e.g. '£45,000' -> '45000').

    Catches any currency symbol (£ $ €) and also bare comma-grouped numbers like '22,500', so a
    figure can't slip through just because the model used the wrong symbol or none at all.
    """
    tokens: set[str] = set()
    for m in re.findall(r"[£$€]\s?(\d[\d,]*)", text):  # symbol-prefixed
        tokens.add(m.replace(",", ""))
    for m in re.findall(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d.,])", text):  # comma-grouped, no symbol
        tokens.add(m.replace(",", ""))
    return tokens


def allowed_figures(ledger: ClientLedger) -> set[str]:
    """The £-figures a report may legitimately state: ledger account values, external funds, amounts."""
    known = {str(int(a.value)) for a in ledger.accounts if a.value is not None}
    known |= {str(int(f.amount)) for f in ledger.external_funds if f.amount is not None}
    known |= {str(int(a)) for a in ledger.amounts}
    return known


def critique_slot(name: str, text: str, ledger: ClientLedger) -> list[str]:
    """Slot-scoped checks for a single generated prose fragment (used by the reflection loop).

    Returns a list of problems phrased as feedback the model can act on.
    """
    problems: list[str] = []
    figures = money_tokens(text)
    if name == "summary":
        # Background must stay high level — no monetary amounts at all.
        if figures:
            problems.append(
                "The Background summary must stay high level and state NO monetary amounts; "
                "remove every figure."
            )
    elif name == "recommendation":
        allowed = allowed_figures(ledger)
        unsourced = sorted(t for t in figures if t not in allowed)
        if unsourced:
            allowed_str = ", ".join(f"£{a}" for a in sorted(allowed)) or "(none)"
            problems.append(
                f"These figures are not provided and must not appear: "
                f"{', '.join('£' + u for u in unsourced)}. "
                f"Use only these exact figures: {allowed_str}. "
                "Do not compute splits, totals, or projected values."
            )
    return problems


def check_internal(report: str) -> list[str]:
    """Minimal check for non-client-facing diagnostic docs (e.g. the reconciliation review sheet).

    The client-report compliance rules (verbatim lines, Tax-iff-disposal, no out-of-ledger figures)
    do not apply to an internal audit view — it deliberately shows stale/rejected values, closed
    accounts and provenance. We only assert that the document was fully rendered.
    """
    if re.search(r"<<\w+>>", report):
        return ["Unfilled <<placeholder>> left in the report."]
    return []


def check_report(report: str, ledger: ClientLedger, *, check_tax: bool = True) -> list[str]:
    """Verify a report against its ledger.

    ``check_tax`` gates the advice-report-specific Tax rules (Tax section iff a disposal; a disposal
    must carry a flag). A different document type that has no Tax section sets it False. All other
    checks are document-type agnostic.
    """
    problems: list[str] = []

    # 1. Verbatim lines present, exactly.
    if FCA_LINE not in report:
        problems.append("FCA authorisation line missing or altered.")
    if RISK_WARNING not in report:
        problems.append("Risk warning missing or altered.")

    # 2. Tax section appears iff there is a disposal (advice report only).
    if check_tax:
        has_tax = "## Tax Implications" in report
        if has_tax != ledger.disposal:
            problems.append(
                f"Tax section presence ({has_tax}) does not match disposal flag ({ledger.disposal})."
            )

    # 3. No leftover placeholders.
    if re.search(r"<<\w+>>", report):
        problems.append("Unfilled <<placeholder>> left in the report.")

    # 4. Human-finalise gaps surface as flags (not hidden, not invented) — but only require a gap's
    #    flag when the section it belongs to is actually present in this document.
    for gap in ledger.gaps:
        section_present = gap.section is None or f"## {gap.section}" in report
        if section_present and gap.field not in report:
            problems.append(f"Gap '{gap.field}' is not surfaced in the report.")
    if check_tax and ledger.disposal and "[FLAG:" not in report:
        problems.append("Disposal report has no flags, but CGT must be flagged.")

    # 5. Holdings table is consistent with the ledger.
    scoped = ledger.scoped_accounts()
    for account in scoped:
        if account.account_id not in report:
            problems.append(f"Scoped account '{account.account_id}' missing from the report.")
    # closed / out-of-scope accounts must not appear
    for account in ledger.accounts:
        if account not in scoped and account.status == "closed" and account.account_id in report:
            problems.append(f"Closed account '{account.account_id}' should not appear.")

    # 6. Every figure in the report traces to a ledger value (no fabricated numbers).
    #    This generally subsumes decoy-leak detection: a decoy figure is not in the ledger, so it is
    #    flagged here — without hard-coding any example's specific numbers.
    known = allowed_figures(ledger)
    for token in money_tokens(report):
        if token not in known:
            problems.append(f"Unsourced figure £{token} in the report (not in the ledger).")

    return problems
