"""Reconciliation: merge db accounts + extracted facts into ONE ClientLedger.

This is the heart of the pipeline and the one place conflicts are resolved. The LLM has already
done the fuzzy work (reading prose, matching it to account_ids); here deterministic house rules
decide the outcome and record an audit trail:

  * recency-wins   — a fresher observed value overrides a stale db snapshot (logged as a Conflict)
  * null -> flag   — a missing value becomes a Gap, never an invented number
  * finalise-gaps  — CGT (on disposal) and ongoing fee rates are always flagged for a human
  * scope          — only report_request's accounts are in scope; closed accounts are excluded
"""

from __future__ import annotations

from datetime import date

from agent_pipeline.extract import ExtractedFacts
from agent_pipeline.funds import available_to_invest
from agent_pipeline.models import (
    Account,
    Action,
    Charges,
    ClientLedger,
    Conflict,
    ExternalFund,
    Gap,
)

_DISPOSAL_WORDS = ("disinvest", "sell", "dispos", "rebalance", "withdraw")


def _is_disposal(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in _DISPOSAL_WORDS)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _apply_observed_value(account: Account, value: float, observed_date: date | None) -> None:
    """Take an observed ("live") value onto an account, marking it approximate and source-tagged."""
    account.value = value
    account.valuation_date = observed_date
    account.approximate = True
    account.value_source = "observed"


def reconcile(accounts: list[Account], facts: ExtractedFacts) -> ClientLedger:
    by_id = {a.account_id: a for a in accounts}

    # --- scope: which accounts the report covers -------------------------------------------
    scope = set(facts.scope_account_ids)
    for a in accounts:
        a.in_scope = (a.account_id in scope) if scope else True

    # --- value reconciliation: recency wins ------------------------------------------------
    conflicts: list[Conflict] = []
    for lv in facts.live_values:
        if lv.value is None or lv.account_id not in by_id:
            continue
        acc = by_id[lv.account_id]
        live_date = _parse_date(lv.as_of)
        db_date = acc.valuation_date
        fresher = live_date and (db_date is None or live_date > db_date)
        if acc.value is not None and fresher and lv.value != acc.value:
            # A fresher observation disagrees with the db snapshot: it wins, and we log why.
            conflicts.append(
                Conflict(
                    field=f"{acc.account_id}.value",
                    chose=f"{lv.value:.0f} @ {live_date} (observed)",
                    over=f"{acc.value:.0f} @ {db_date} (db snapshot)",
                    rule="most-recent-valuation-wins",
                )
            )
            _apply_observed_value(acc, lv.value, live_date)
        elif acc.value is None and lv.value is not None:
            # The db had no figure at all; take the observed one (nothing to conflict with).
            _apply_observed_value(acc, lv.value, live_date)

    # --- discovered accounts from unrecognised documents (kept, but flagged for review) ----
    review_gaps: list[Gap] = []
    for index, found in enumerate(facts.discovered_accounts, start=1):
        aid = found.account_id or f"NEW-{index}"
        if aid in by_id:
            continue  # the document is talking about an account we already know
        account = Account(
            account_id=aid,
            owner=found.owner or "unknown",
            type=found.type or "unknown",
            value=found.value,
            valuation_date=_parse_date(found.as_of),
            in_scope=True,
            approximate=True,
            value_source="unknown",
        )
        accounts.append(account)
        by_id[aid] = account
        review_gaps.append(
            Gap(
                field=f"{aid} (discovered)",
                reason="account found in an unrecognised document; verify before relying on it",
                section="Background & Objectives",
                review=True,
            )
        )
    for note in facts.unmapped:
        review_gaps.append(
            Gap(field="unreviewed material", reason=note, section="Background & Objectives", review=True)
        )

    # --- disposal flag (drives the Tax section) --------------------------------------------
    disposal = bool(facts.selling) or any(_is_disposal(a) for a in facts.actions)

    # --- gaps: things a human must finalise, or values we don't have -----------------------
    gaps: list[Gap] = []
    for a in accounts:
        if a.in_scope and a.status != "closed" and a.value is None:
            gaps.append(
                Gap(
                    field=f"{a.account_id} value",
                    reason="balance not captured at snapshot; to be confirmed",
                    section="Background & Objectives",
                )
            )
    gaps.append(Gap(field="platform charge", reason="ongoing platform charge to confirm", section="Fees & Charges"))
    gaps.append(Gap(field="advice charge", reason="ongoing advice charge to confirm", section="Fees & Charges"))
    if disposal:
        gaps.append(
            Gap(
                field="capital gains tax",
                reason="liability on the disposal to be confirmed by adviser",
                section="Tax Implications",
            )
        )
    gaps.extend(review_gaps)  # discovered accounts / unmapped material from unrecognised documents

    actions = [Action(text=t, is_disposal=_is_disposal(t)) for t in facts.actions]
    external = [
        ExternalFund(label=f.label, amount=f.amount, kind=f.kind, note=f.note)
        for f in facts.external_funds
    ]

    # Funds calculator (deterministic): make the available-to-invest total a sourced figure so the
    # recommendation can state it and verification accepts it.
    amounts = list(facts.investment_amounts)
    investable = available_to_invest(external)
    if investable is not None and investable not in amounts:
        amounts.append(investable)

    return ClientLedger(
        client=facts.client_label or "the client",
        risk_profile=facts.risk_profile,
        accounts=accounts,
        actions=actions,
        external_funds=external,
        disposal=disposal,
        charges=Charges(initial=facts.initial_charge),
        gaps=gaps,
        objectives=facts.objectives,
        amounts=amounts,
        guidance=facts.guidance,
        conflicts=conflicts,
    )
