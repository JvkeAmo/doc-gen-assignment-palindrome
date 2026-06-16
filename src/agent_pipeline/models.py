"""The ClientLedger: the canonical, reconciled intermediate the pipeline is built around.

Source files (db, meeting notes, report request, image) are extracted into tagged claims,
those claims are reconciled into ONE ``ClientLedger``, and every report section is then
generated from a slice of the ledger rather than from raw files. Keeping a single typed
object here means:

  * conflict resolution happens in one auditable place (see ``Conflict``),
  * generation can only ever see clean, trusted facts (no decoys, no raw conflicts),
  * verification has a concrete contract to assert against.

The schema is kept deliberately small: every field is something reconciliation populates from the
sources and the verifier can later assert the report against.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Account(BaseModel):
    """One account covered by (or relevant to) the report, after reconciliation.

    A joint account is represented once here, even though the source db lists it under
    every holder. ``value is None`` means no trustworthy figure was found and a matching
    ``Gap`` should exist instead of an invented number.
    """

    account_id: str
    owner: str
    type: str
    platform: str | None = None
    value: float | None = None
    currency: str = "GBP"
    valuation_date: date | None = None
    status: str = "open"
    in_scope: bool = True
    approximate: bool = False  # True when the chosen value came from a vague "live" figure
    value_source: str | None = None  # provenance of the value we kept (e.g. "meeting_notes")


class Action(BaseModel):
    """A recommended action from the adviser's instruction / meeting note."""

    text: str
    refs: list[str] = Field(default_factory=list)  # account_ids involved
    is_disposal: bool = False
    amount: float | None = None


class ExternalFund(BaseModel):
    """Money that is not (yet) an account: an inheritance, business-sale proceeds, etc.

    ``kind`` is how the funds calculator treats it:
      * "available"  — an inflow that can be invested now (inheritance, completion payment received);
      * "contingent" — a future/uncertain inflow not yet received (an earnout) — excluded from totals;
      * "committed"  — an amount already earmarked for an outflow (a loan repayment) — subtracted.
    """

    label: str
    amount: float | None = None
    kind: str = "available"
    note: str | None = None


class Charges(BaseModel):
    """Fee structure. ``None`` on an ongoing rate means it must be flagged, never invented."""

    initial: str | None = None  # usually known from report_request, e.g. "0%"
    platform_ongoing: str | None = None
    advice_ongoing: str | None = None


class Gap(BaseModel):
    """A value that a human must finalise, or a figure missing from the sources.

    Rendered as a visible marker in the report so nothing is silently hidden.
    """

    field: str
    reason: str
    section: str | None = None  # which report section it surfaces in
    review: bool = False  # raised from an unrecognised document — surface prominently for review

    def marker(self) -> str:
        return f"[FLAG: {self.field} — {self.reason}]"


class Conflict(BaseModel):
    """An audit-trail entry: when sources disagreed, what we chose and why."""

    field: str
    chose: str
    over: str
    rule: str


class ClientLedger(BaseModel):
    """The single reconciled source of truth a report is generated from."""

    client: str
    risk_profile: str | None = None
    accounts: list[Account] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    external_funds: list[ExternalFund] = Field(default_factory=list)
    disposal: bool = False
    charges: Charges = Field(default_factory=Charges)
    gaps: list[Gap] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)  # high-level circumstances (no amounts)
    amounts: list[float] = Field(default_factory=list)  # headline figures stated in the sources
    guidance: list[str] = Field(default_factory=list)  # per-client notes (fde_notes "## This client")
    conflicts: list[Conflict] = Field(default_factory=list)

    def scoped_accounts(self) -> list[Account]:
        """Accounts actually covered by this report (closed/out-of-scope excluded)."""
        return [a for a in self.accounts if a.in_scope and a.status != "closed"]

    def objectives_text(self) -> str:
        return "; ".join(self.objectives)
