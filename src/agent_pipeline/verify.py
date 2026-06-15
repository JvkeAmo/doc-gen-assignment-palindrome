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

# Figures from the general/decoy documents that must never reach a report.
DECOY_FIGURES = ["312,000", "505,000", "515,000", "775,000", "525,000", "785,000", "312000"]


def _money_tokens(text: str) -> set[str]:
    """All £-amounts in the text, normalised to digits-only (e.g. '£45,000' -> '45000')."""
    return {m.replace(",", "") for m in re.findall(r"£\s?([\d,]+)", text)}


def check_report(report: str, ledger: ClientLedger) -> list[str]:
    problems: list[str] = []

    # 1. Verbatim lines present, exactly.
    if FCA_LINE not in report:
        problems.append("FCA authorisation line missing or altered.")
    if RISK_WARNING not in report:
        problems.append("Risk warning missing or altered.")

    # 2. Tax section appears iff there is a disposal.
    has_tax = "## Tax Implications" in report
    if has_tax != ledger.disposal:
        problems.append(
            f"Tax section presence ({has_tax}) does not match disposal flag ({ledger.disposal})."
        )

    # 3. No leftover placeholders.
    if re.search(r"<<\w+>>", report):
        problems.append("Unfilled <<placeholder>> left in the report.")

    # 4. No decoy figures.
    for fig in DECOY_FIGURES:
        if fig in report:
            problems.append(f"Decoy/general figure '{fig}' leaked into the report.")

    # 5. Human-finalise gaps surface as flags (not hidden, not invented).
    for gap in ledger.gaps:
        if gap.field not in report:
            problems.append(f"Gap '{gap.field}' is not surfaced in the report.")
    if ledger.disposal and "[FLAG:" not in report:
        problems.append("Disposal report has no flags, but CGT must be flagged.")

    # 6. Holdings table is consistent with the ledger.
    scoped = ledger.scoped_accounts()
    for account in scoped:
        if account.account_id not in report:
            problems.append(f"Scoped account '{account.account_id}' missing from the report.")
    # closed / out-of-scope accounts must not appear
    for account in ledger.accounts:
        if account not in scoped and account.status == "closed" and account.account_id in report:
            problems.append(f"Closed account '{account.account_id}' should not appear.")

    # 7. Every £-figure in the report traces to a ledger value (no fabricated numbers).
    known = {str(int(a.value)) for a in ledger.accounts if a.value is not None}
    known |= {str(int(f.amount)) for f in ledger.external_funds if f.amount is not None}
    known |= {str(int(a)) for a in ledger.amounts}
    for token in _money_tokens(report):
        if token not in known:
            problems.append(f"Unsourced figure £{token} in the report (not in the ledger).")

    return problems
