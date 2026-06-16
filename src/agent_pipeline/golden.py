"""Golden-ledger comparison: evaluate the extraction+reconciliation against a hand-authored truth.

A "golden ledger" is the expected reconciled facts for an example client — the stable, critical
fields only (values, dates, scope, disposal, conflicts, external funds, gaps), since free-text fields
(action/objective/guidance wording) vary with the model. This evaluates the *heart* of the pipeline
deterministically; it complements the report-level rule checks in ``verify.py``.

These goldens live under ``eval/golden/`` — test fixtures, not production logic, so they are not the
overfitting the brief warns against (that would be baking example values into prompts/checks that run
on the held-out set).
"""

from __future__ import annotations

from agent_pipeline.models import ClientLedger


def _round(value: float | None) -> int | None:
    return None if value is None else round(value)


def compare_ledger(led: ClientLedger, golden: dict) -> list[str]:
    """Return a list of mismatches between a produced ledger and the golden expectation."""
    problems: list[str] = []

    if "disposal" in golden and led.disposal != golden["disposal"]:
        problems.append(f"disposal {led.disposal} != expected {golden['disposal']}")

    if golden.get("charges_initial") is not None and led.charges.initial != golden["charges_initial"]:
        problems.append(f"initial charge {led.charges.initial!r} != {golden['charges_initial']!r}")

    if "scope" in golden:
        scoped = {a.account_id for a in led.scoped_accounts()}
        if scoped != set(golden["scope"]):
            problems.append(f"scope {sorted(scoped)} != expected {sorted(golden['scope'])}")

    by_id = {a.account_id: a for a in led.accounts}
    for aid, exp in golden.get("accounts", {}).items():
        account = by_id.get(aid)
        if account is None:
            problems.append(f"account {aid} missing from ledger")
            continue
        if _round(account.value) != exp.get("value"):
            problems.append(f"{aid} value {_round(account.value)} != expected {exp.get('value')}")
        actual_date = account.valuation_date.isoformat() if account.valuation_date else None
        if actual_date != exp.get("valuation_date"):
            problems.append(f"{aid} valuation_date {actual_date} != expected {exp.get('valuation_date')}")
        if account.status != exp.get("status"):
            problems.append(f"{aid} status {account.status!r} != expected {exp.get('status')!r}")

    conflict_ids = {c.field.split(".")[0] for c in led.conflicts}
    for cid in golden.get("conflicts", []):
        if cid not in conflict_ids:
            problems.append(f"expected recency conflict on {cid} not found")

    for fund in golden.get("external_funds", []):
        if not any(
            _round(f.amount) == fund["amount"] and f.kind == fund["kind"]
            for f in led.external_funds
        ):
            problems.append(f"expected external fund {fund} not found")

    if "available_to_invest" in golden:
        from agent_pipeline.funds import available_to_invest

        actual = available_to_invest(led.external_funds)
        if actual != golden["available_to_invest"]:
            problems.append(
                f"available_to_invest {actual} != expected {golden['available_to_invest']}"
            )

    gap_text = " | ".join(g.field for g in led.gaps)
    for needle in golden.get("gaps_include", []):
        if needle not in gap_text:
            problems.append(f"expected gap containing '{needle}' not found")

    return problems
