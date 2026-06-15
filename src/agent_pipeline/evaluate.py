"""Evaluation harness: check generated reports against their ledgers.

We are given no expected outputs, so "correct" is defined by the deterministic rules in
``verify.check_report`` (verbatim lines present, Tax section iff disposal, no decoy or unsourced
figures, human-finalise gaps shown as flags, holdings table consistent with the ledger).

This runs those checks over the already-generated artifacts in ``outputs/`` and
``outputs/ledgers/`` and prints a pass/fail summary. It exits non-zero if any client fails, so it
can gate CI. Generate first with:

    python -m agent_pipeline.generate --client <name> --ledger-dir outputs/ledgers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_pipeline.models import ClientLedger
from agent_pipeline.verify import check_report


def evaluate(output_dir: Path, ledger_dir: Path) -> int:
    ledgers = sorted(ledger_dir.glob("*.json"))
    if not ledgers:
        print(f"No ledgers found in {ledger_dir}. Generate with --ledger-dir first.")
        return 1

    total_problems = 0
    for ledger_path in ledgers:
        name = ledger_path.stem
        report_path = output_dir / f"{name}.md"
        if not report_path.exists():
            print(f"[!] {name}: report missing ({report_path})")
            total_problems += 1
            continue
        ledger = ClientLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
        problems = check_report(report_path.read_text(encoding="utf-8"), ledger)
        if problems:
            print(f"[FAIL] {name}: {len(problems)} issue(s)")
            for problem in problems:
                print(f"        - {problem}")
            total_problems += len(problems)
        else:
            print(f"[PASS] {name}")

    print(f"\n{len(ledgers)} client(s) checked, {total_problems} issue(s) total.")
    return 1 if total_problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated reports against their ledgers.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--ledger-dir", type=Path, default=Path("outputs/ledgers"))
    args = parser.parse_args()
    sys.exit(evaluate(args.output_dir, args.ledger_dir))


if __name__ == "__main__":
    main()
