"""Funds calculator: the deterministic arithmetic the LLM should not be trusted with.

Division of labour: the LLM does the *fuzzy* part during extraction — classifying each external fund
as ``available`` (investable now), ``contingent`` (not yet received, e.g. an earnout), or
``committed`` (earmarked outflow, e.g. a loan repayment). This module does the *arithmetic*:

    available_to_invest = sum(available inflows) - sum(committed outflows)      # contingent excluded

The result is written into the ledger as a sourced figure, so the recommendation can state it and
verification accepts it — unlike a number the model computes in prose, which would be unsourced
(and often wrong).
"""

from __future__ import annotations

from agent_pipeline.models import ExternalFund


def available_to_invest(funds: list[ExternalFund]) -> float | None:
    """Investable-now total, or None when there are no external funds (not applicable)."""
    if not funds:
        return None
    inflow = sum(f.amount or 0 for f in funds if f.kind == "available")
    committed = sum(f.amount or 0 for f in funds if f.kind == "committed")
    return inflow - committed


def funds_breakdown(funds: list[ExternalFund]) -> list[str]:
    """Human-readable lines describing each fund and the computed available-to-invest."""
    lines: list[str] = []
    for f in funds:
        amount = f"£{f.amount:,.0f}" if f.amount is not None else "amount to confirm"
        lines.append(f"{f.label}: {amount} [{f.kind}]")
    total = available_to_invest(funds)
    if total is not None:
        # Be precise: this is NEW external money. It deliberately excludes proceeds from disinvesting
        # existing accounts (those are separate, and reported via the account values).
        lines.append(f"=> new money available to invest now (excludes any disposal proceeds): £{total:,.0f}")
    return lines
