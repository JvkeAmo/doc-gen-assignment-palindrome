"""Evaluation harness: two complementary checks over the generated artifacts.

1. **Report rules** (`verify.check_report`) — does the final report satisfy the invariants we know
   must hold (verbatim lines, Tax iff disposal, gaps flagged, every figure sourced, table consistent).
2. **Golden ledgers** (`golden.compare_ledger`) — does extraction+reconciliation produce the expected
   reconciled facts for each example (the deterministic heart). Goldens live in ``eval/golden/``.

Both run over the artifacts in ``outputs/`` and ``outputs/ledgers/`` and exit non-zero on any failure,
so this can gate CI. Generate first with:

    python -m agent_pipeline.generate --client <name> --ledger-dir outputs/ledgers
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from agent_pipeline.golden import compare_ledger
from agent_pipeline.models import ClientLedger
from agent_pipeline.verify import check_report


def evaluate(output_dir: Path, ledger_dir: Path, golden_dir: Path, judge: bool = False) -> int:
    ledgers = sorted(ledger_dir.glob("*.json"))
    if not ledgers:
        print(f"No ledgers found in {ledger_dir}. Generate with --ledger-dir first.")
        return 1

    # Load every client's ledger + report up front so the (slow) judge calls can run concurrently.
    loaded = []  # (name, ledger, report)
    for ledger_path in ledgers:
        name = ledger_path.stem
        ledger = ClientLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
        report_path = output_dir / f"{name}.md"
        report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        loaded.append((name, ledger, report))

    # Run all LLM-judge calls in parallel (eval-only, soft metric). pool.map keeps results aligned
    # with `loaded`, so the printed output below stays in order.
    judge_scores: dict[str, dict] = {}
    if judge:
        from agent_pipeline.judge import judge_report
        from agent_pipeline.llm import build_client, model_name  # local import: only when judging

        judge_client, judge_model = build_client(), model_name()
        with ThreadPoolExecutor(max_workers=len(loaded)) as pool:
            scored = pool.map(
                lambda item: judge_report(judge_client, judge_model, item[2], item[1]) if item[2] else {},
                loaded,
            )
        judge_scores = {name: result for (name, _ledger, _report), result in zip(loaded, scored)}

    total_problems = 0
    for name, ledger, report in loaded:
        problems: list[str] = []

        # 1. report rules (hard pass/fail)
        if report:
            problems += [f"report: {p}" for p in check_report(report, ledger)]
        else:
            problems.append(f"report: missing ({output_dir / f'{name}.md'})")

        # 2. golden ledger (hard pass/fail), if one exists for this client
        golden_path = golden_dir / f"{name}.json"
        if golden_path.exists():
            golden = json.loads(golden_path.read_text(encoding="utf-8"))
            problems += [f"ledger: {p}" for p in compare_ledger(ledger, golden)]

        if problems:
            print(f"[FAIL] {name}: {len(problems)} issue(s)")
            for problem in problems:
                print(f"        - {problem}")
            total_problems += len(problems)
        else:
            print(f"[PASS] {name}")

        # 3. LLM-judge quality scores (soft metric, reported not gated)
        for dimension, result in judge_scores.get(name, {}).items():
            if isinstance(result, dict):
                print(f"        ~ {dimension}: {result.get('score')}/5 — {result.get('reason', '')}")

    print(f"\n{len(ledgers)} client(s) checked, {total_problems} issue(s) total.")
    return 1 if total_problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated reports and ledgers.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--ledger-dir", type=Path, default=Path("outputs/ledgers"))
    parser.add_argument("--golden-dir", type=Path, default=Path("eval/golden"))
    parser.add_argument("--judge", action="store_true", help="also run the LLM-judge quality scores (slow)")
    args = parser.parse_args()
    load_dotenv()  # so --judge can read the API key from .env
    sys.exit(evaluate(args.output_dir, args.ledger_dir, args.golden_dir, args.judge))


if __name__ == "__main__":
    main()
