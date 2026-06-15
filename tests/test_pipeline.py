"""Deterministic unit tests for the non-LLM parts of the pipeline.

These cover the rules that matter most and need no model: db de-duplication, recency-wins
reconciliation, and the verification checks. Run with ``uv run pytest``.
"""

from agent_pipeline.extract import ExtractedFacts, LiveValue, merge_facts, parse_db
from agent_pipeline.models import Account, ClientLedger, Gap
from agent_pipeline.reconcile import reconcile
from agent_pipeline.verify import FCA_LINE, RISK_WARNING, check_report

DB_JSON = """
{
  "snapshot_date": "2026-04-30",
  "holders": {
    "client": {"name": "A", "accounts": [
      {"account_id": "ISA-A", "platform": "P", "type": "ISA", "owner": "A", "status": "open", "value": 61000.0, "currency": "GBP", "valuation_date": "2026-04-30"},
      {"account_id": "GIA-J", "platform": "P", "type": "GIA", "owner": "Joint", "status": "open", "value": 40000.0, "currency": "GBP", "valuation_date": "2026-03-15"}
    ]},
    "partner": {"name": "B", "accounts": [
      {"account_id": "GIA-J", "platform": "P", "type": "GIA", "owner": "Joint", "status": "open", "value": 40000.0, "currency": "GBP", "valuation_date": "2026-03-15"}
    ]}
  }
}
"""


def test_parse_db_dedupes_joint_account():
    accounts, snapshot = parse_db(DB_JSON)
    ids = [a.account_id for a in accounts]
    assert ids.count("GIA-J") == 1  # joint listed under both holders, kept once
    assert str(snapshot) == "2026-04-30"


def test_reconcile_recency_wins_and_logs_conflict():
    accounts, _ = parse_db(DB_JSON)
    facts = ExtractedFacts(
        selling=True,
        scope_account_ids=["ISA-A", "GIA-J"],
        live_values=[LiveValue(account_id="GIA-J", value=45000.0, as_of="2026-05-14")],
    )
    ledger = reconcile(accounts, facts)
    gia = next(a for a in ledger.accounts if a.account_id == "GIA-J")
    assert gia.value == 45000.0  # fresher meeting figure beats stale db
    assert gia.approximate is True
    assert ledger.disposal is True
    assert any(c.field == "GIA-J.value" for c in ledger.conflicts)


def test_verify_catches_missing_verbatim_and_unsourced_figure():
    ledger = ClientLedger(
        client="A",
        disposal=False,
        accounts=[Account(account_id="ISA-A", owner="A", type="ISA", value=61000.0)],
    )
    bad_report = "# Report\n\nYour ISA-A is worth £61,000 and we project £99,000 next year."
    problems = check_report(bad_report, ledger)
    assert any("FCA" in p for p in problems)
    assert any("Risk warning" in p for p in problems)
    assert any("99000" in p for p in problems)  # invented figure flagged
    assert not any("61000" in p for p in problems)  # sourced figure is fine


def test_extracted_facts_coerces_scalars_to_lists():
    # Models sometimes return a bare string where a list is expected; it must not blow up.
    facts = ExtractedFacts.model_validate(
        {"guidance": "handle sensitively", "actions": "do one thing", "live_values": {"account_id": "X", "value": 1}}
    )
    assert facts.guidance == ["handle sensitively"]
    assert facts.actions == ["do one thing"]
    assert len(facts.live_values) == 1 and facts.live_values[0].account_id == "X"


def test_merge_facts_unions_lists_and_keeps_first_scalar():
    a = ExtractedFacts(selling=True, scope_account_ids=["A"], actions=["x"])
    b = ExtractedFacts(risk_profile="4", scope_account_ids=["A", "B"], actions=["y"])
    merged = merge_facts([a, b])
    assert merged.selling is True
    assert merged.risk_profile == "4"
    assert merged.scope_account_ids == ["A", "B"]  # union, de-duped
    assert merged.actions == ["x", "y"]


def test_verify_passes_a_clean_report():
    ledger = ClientLedger(
        client="A",
        disposal=False,
        accounts=[Account(account_id="ISA-A", owner="A", type="ISA", value=61000.0)],
        gaps=[Gap(field="platform charge", reason="to confirm", section="Fees & Charges")],
    )
    report = (
        "# Investment Advice Report\n\n## Introduction\n\n"
        f"{FCA_LINE}\n\n## Background & Objectives\n\n"
        "| Account | Owner | Type | Value |\n|---|---|---|---|\n"
        "| ISA-A | A | ISA | £61,000 |\n\n## Fees & Charges\n\n"
        "[FLAG: platform charge — to confirm]\n\n## Conclusion\n\n"
        f"{RISK_WARNING}\n"
    )
    assert check_report(report, ledger) == []
