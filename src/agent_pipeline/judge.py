"""LLM-as-judge: an EVAL-ONLY quality metric for the generated prose.

The deterministic checks in ``verify.py`` cover *compliance* (figures sourced, verbatim lines, gaps
flagged). They cannot judge *quality* — tone, sensitivity, whether the Background stays high level,
whether per-client guidance was applied. This module asks a model to score those dimensions against
the ledger (the ground truth), for measuring/comparing pipeline versions and to inform prompt tuning.

Deliberately NOT used at inference: the runtime guard is the fast, deterministic ``critique_slot``.
The judge is slower, costlier and a bit flaky — better where scores are read by a human and noise
averages out. Run via ``evaluate --judge``.
"""

from __future__ import annotations

from openai import OpenAI

from agent_pipeline.funds import available_to_invest
from agent_pipeline.llm import complete, loads_json
from agent_pipeline.models import ClientLedger

_JUDGE_SYSTEM = (
    "You are a careful reviewer of UK financial advice reports. Score each dimension from 1 (poor) "
    "to 5 (excellent) and give a one-line reason. Return ONLY a JSON object."
)


def _ledger_summary(ledger: ClientLedger) -> str:
    lines = [f"Client: {ledger.client}", f"Disposal: {ledger.disposal}"]
    for a in ledger.scoped_accounts():
        value = "to confirm" if a.value is None else f"£{a.value:,.0f}"
        lines.append(f"- account {a.account_id} {a.type} ({a.owner}): {value}")
    if ledger.amounts:
        lines.append("Stated investment amounts: " + ", ".join(f"£{a:,.0f}" for a in ledger.amounts))
    for f in ledger.external_funds:
        amount = f"£{f.amount:,.0f}" if f.amount is not None else "amount to confirm"
        lines.append(f"- external fund: {f.label} {amount} [{f.kind}]")
    investable = available_to_invest(ledger.external_funds)
    if investable is not None:
        lines.append(f"New money available to invest now (excludes disposal proceeds): £{investable:,.0f}")
    if ledger.actions:
        lines.append("Agreed actions: " + "; ".join(a.text for a in ledger.actions))
    if ledger.guidance:
        lines.append("Sensitive guidance: " + "; ".join(ledger.guidance))
    return "\n".join(lines)


def judge_report(client: OpenAI, model: str, report: str, ledger: ClientLedger) -> dict:
    """Return {dimension: {"score": int, "reason": str}} for the report's prose quality."""
    prompt = (
        "Reference facts (the ground truth the report must match):\n"
        f"{_ledger_summary(ledger)}\n\n"
        "Report to review:\n"
        f"{report}\n\n"
        "Score these dimensions as JSON {dimension: {\"score\": 1-5, \"reason\": str}}:\n"
        '  "faithfulness": do all stated facts and figures match the reference (no contradictions or '
        "invented numbers)?\n"
        '  "background_altitude": does the Background & Objectives PROSE stay high level, with no '
        "specific transaction amounts (top-ups, sale proceeds, tax)? The accounts table "
        "(Account/Owner/Type/Value) is expected there and is fine — judge only the prose;\n"
        '  "sensitivity": are any sensitive circumstances handled tactfully?\n'
        '  "clarity": is it clear, professional British-English advice?'
    )
    raw = complete(client, model, prompt, system=_JUDGE_SYSTEM, as_json=True)
    try:
        return loads_json(raw)
    except Exception:  # noqa: BLE001 - judging is best-effort
        return {}
