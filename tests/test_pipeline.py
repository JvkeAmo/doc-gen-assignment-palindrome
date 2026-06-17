"""Deterministic unit tests for the non-LLM parts of the pipeline.

These cover the rules that matter most and need no model: db de-duplication, recency-wins
reconciliation, and the verification checks. Run with ``uv run pytest``.
"""

from agent_pipeline.extract import ExtractedFacts, LiveValue, merge_facts, parse_db
from agent_pipeline.models import Account, Charges, ClientLedger, Gap
from agent_pipeline.reconcile import reconcile
from agent_pipeline.render import fill_placeholder, generate_with_reflection
from agent_pipeline.verify import FCA_LINE, RISK_WARNING, check_report, critique_slot


class _StubClient:
    """Minimal stand-in for the OpenAI client: returns queued responses in order."""

    def __init__(self, responses):
        outs = list(responses)

        class _Completions:
            def create(self, **_kwargs):
                content = outs.pop(0)
                message = type("M", (), {"content": content})
                choice = type("C", (), {"message": message})
                return type("R", (), {"choices": [choice]})

        self.chat = type("Chat", (), {"completions": _Completions()})()

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


def test_critique_slot_flags_summary_figures_and_unsourced_recommendation():
    ledger = ClientLedger(
        client="A",
        accounts=[Account(account_id="GIA-J", owner="Joint", type="GIA", value=45000.0)],
    )
    # summary must carry no figures at all
    assert critique_slot("summary", "Retired clients seeking growth.", ledger) == []
    assert critique_slot("summary", "They hold about £45,000.", ledger)
    # recommendation may cite a sourced value, but not a computed split
    assert critique_slot("recommendation", "Disinvest the GIA (£45,000).", ledger) == []
    assert critique_slot("recommendation", "Each ISA receives £22,500.", ledger)


def test_reflection_retries_until_clean():
    ledger = ClientLedger(
        client="A",
        accounts=[Account(account_id="GIA-J", owner="Joint", type="GIA", value=45000.0)],
    )
    client = _StubClient(["Each ISA receives £22,500.", "Disinvest the GIA (£45,000)."])
    out = generate_with_reflection("recommendation", "BASE", ledger, client, "m", None, max_retries=2)
    assert "£22,500" not in out  # the bad first attempt was rejected
    assert "£45,000" in out  # the clean revision was accepted
    assert "[FLAG" not in out


def test_reflection_flags_when_unresolved():
    ledger = ClientLedger(
        client="A",
        accounts=[Account(account_id="GIA-J", owner="Joint", type="GIA", value=45000.0)],
    )
    client = _StubClient(["£99,999 each", "£99,999 each", "£99,999 each"])
    out = generate_with_reflection("recommendation", "BASE", ledger, client, "m", None, max_retries=2)
    assert "[FLAG: section needs review" in out  # shipped with a visible flag, not hidden


def test_funds_calculator_nets_committed_and_excludes_contingent():
    from agent_pipeline.funds import available_to_invest
    from agent_pipeline.models import ExternalFund

    funds = [
        ExternalFund(label="completion", amount=850000, kind="available"),
        ExternalFund(label="earnout", amount=400000, kind="contingent"),
        ExternalFund(label="bridging", amount=200000, kind="committed"),
    ]
    assert available_to_invest(funds) == 650000  # 850k available - 200k committed; earnout excluded
    assert available_to_invest([]) is None


def test_render_fees_fills_config_template():
    # The fees wording now lives in the config template; the renderer only supplies the
    # conditional clauses. The assembled result must match the previous code-built string exactly.
    ledger = ClientLedger(
        client="A",
        charges=Charges(initial="0%"),
        gaps=[
            Gap(field="platform charge", reason="ongoing platform charge to confirm", section="Fees & Charges"),
            Gap(field="advice charge", reason="ongoing advice charge to confirm", section="Fees & Charges"),
        ],
    )
    spec = {
        "source": "render:fees",
        "template": "The ongoing charges that apply are the platform charge levied by the platform "
        "and our ongoing advice charge.{initial_charge}{fee_flags}",
    }
    out = fill_placeholder("fees", spec, ledger, None, "m", "")
    assert out == (
        "The ongoing charges that apply are the platform charge levied by the platform and our "
        "ongoing advice charge. The initial charge on this recommendation is 0%. "
        "[FLAG: platform charge — ongoing platform charge to confirm] "
        "[FLAG: advice charge — ongoing advice charge to confirm]"
    )


def test_render_cgt_fills_config_template():
    ledger = ClientLedger(
        client="A",
        disposal=True,
        gaps=[
            Gap(
                field="capital gains tax",
                reason="liability on the disposal to be confirmed by adviser",
                section="Tax Implications",
            )
        ],
    )
    spec = {
        "source": "render:cgt_statement",
        "template": "The recommended disposal may give rise to a capital gains tax liability, which "
        "would be assessed against your annual exempt amount.{cgt_flag}",
    }
    out = fill_placeholder("cgt_statement", spec, ledger, None, "m", "")
    assert out == (
        "The recommended disposal may give rise to a capital gains tax liability, which would be "
        "assessed against your annual exempt amount.\n\n"
        "[FLAG: capital gains tax — liability on the disposal to be confirmed by adviser]"
    )


def _portfolio_style_report():
    # A non-advice doc: verbatim lines + holdings table, but no Tax/Fees sections.
    return (
        "# Portfolio Review Summary\n\n## Introduction\n\n"
        f"{FCA_LINE}\n\n## Background & Objectives\n\n"
        "| Account | Owner | Type | Value |\n|---|---|---|---|\n"
        "| ISA-A | A | ISA | £61,000 |\n\n## Conclusion\n\n"
        f"{RISK_WARNING}\n"
    )


def test_check_report_tax_rules_can_be_disabled_for_other_doc_types():
    ledger = ClientLedger(
        client="A",
        disposal=True,
        accounts=[Account(account_id="ISA-A", owner="A", type="ISA", value=61000.0)],
    )
    report = _portfolio_style_report()
    # The advice-report default flags a disposal with no Tax section...
    assert any("Tax section" in p for p in check_report(report, ledger))
    # ...but a doc type that declares no Tax section passes.
    assert check_report(report, ledger, check_tax=False) == []


def test_check_report_only_requires_gaps_for_present_sections():
    ledger = ClientLedger(
        client="A",
        disposal=False,
        accounts=[Account(account_id="ISA-A", owner="A", type="ISA", value=61000.0)],
        gaps=[Gap(field="platform charge", reason="to confirm", section="Fees & Charges")],
    )
    # The doc has no "## Fees & Charges" section, so the fee gap must not be required here.
    assert check_report(_portfolio_style_report(), ledger, check_tax=False) == []


def test_is_disposal_covers_synonyms_but_not_transfers():
    from agent_pipeline.reconcile import _is_disposal

    assert _is_disposal("liquidate the holdings")
    assert _is_disposal("encash the bond")
    assert _is_disposal("disinvest the joint GIA in full")
    # moving cash into an ISA is NOT a disposal (client_01 must not get a Tax section)
    assert not _is_disposal("transfer £20,000 from H-CASH-01 to H-ISA-01")


def test_recommendation_context_separates_new_money_from_disposal_proceeds():
    from agent_pipeline.models import Action, ExternalFund
    from agent_pipeline.render import _recommendation_context

    ledger = ClientLedger(
        client="A",
        disposal=True,
        accounts=[Account(account_id="GIA-J", owner="Joint", type="GIA", value=38000.0)],
        external_funds=[ExternalFund(label="inheritance", amount=120000, kind="available")],
        amounts=[120000.0],
        actions=[Action(text="disinvest the joint GIA in full", is_disposal=True)],
    )
    context = _recommendation_context(ledger)
    # the £120k is framed as NEW money, not "the investable total"...
    assert "New money available to invest" in context
    # ...and the model is told disposal proceeds are a separate component, not to be summed.
    assert "two separate sources" in context.lower() or "two separate" in context.lower()
    assert "do not add them into a single total" in context.lower()
    from pathlib import Path

    from agent_pipeline.triage import Role, classify

    assert classify(Path("mystery_letter.txt")) is Role.UNKNOWN  # unrecognised -> exploratory pass
    assert classify(Path("fde_notes.md")) is Role.GUIDANCE  # known guidance unchanged
    assert classify(Path("template_spec.md")) is None  # ignored unchanged


def test_unknown_document_discovers_account_and_flags_unmapped():
    import json as _json

    from agent_pipeline.extract import extract_facts
    from agent_pipeline.triage import Role

    accounts, _ = parse_db(DB_JSON)
    explore_json = _json.dumps(
        {
            "discovered_accounts": [
                {"account_id": "OFFSHORE-1", "type": "Offshore Bond", "owner": "A", "value": 50000,
                 "as_of": "2026-05-01"}
            ],
            "unmapped": ["mentions a possible trust arrangement"],
        }
    )
    client = _StubClient([explore_json])
    facts = extract_facts(client, "m", accounts, {Role.UNKNOWN: "some novel document text"})
    ledger = reconcile(accounts, facts)

    discovered = next(a for a in ledger.accounts if a.account_id == "OFFSHORE-1")
    assert discovered.value_source == "unknown" and discovered.in_scope  # kept, provenance-tagged
    assert any(g.review and "OFFSHORE-1" in g.field for g in ledger.gaps)  # flagged for review
    assert any(g.review and g.reason == "mentions a possible trust arrangement" for g in ledger.gaps)


def test_review_flags_surface_in_holdings_table():
    from agent_pipeline.render import render_holdings_table

    ledger = ClientLedger(
        client="A",
        accounts=[Account(account_id="ISA-A", owner="A", type="ISA", value=61000.0)],
        gaps=[Gap(field="unreviewed material", reason="a trust", section="Background & Objectives", review=True)],
    )
    table = render_holdings_table(ledger)["table"]
    assert "[FLAG: unreviewed material — a trust]" in table  # surfaced, not buried in the ledger


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
