"""Generate an advice report for a client.

Pipeline: triage the source files, extract facts (db deterministically, each prose source via its
own focused LLM call), reconcile them into a single ClientLedger, then generate each section from a
slice of that ledger and assemble the document. Every run writes telemetry to outputs/runs/.

Usage:
    python -m agent_pipeline.generate --client client_01_clean
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from agent_pipeline.extract import extract_facts, parse_db
from agent_pipeline.llm import build_client, model_name
from agent_pipeline.models import ClientLedger
from agent_pipeline.ocr import read_image_text
from agent_pipeline.reconcile import reconcile
from agent_pipeline.render import fill_placeholder
from agent_pipeline.runlog import RunRecorder
from agent_pipeline.triage import Role, triage_folder
from agent_pipeline.verify import check_report
from document_formatter.formatting import format_document
from document_formatter.loading import read_file


def read_sources(grouped: dict[Role, list[Path]], image_text: dict[str, str]) -> dict[Role, str]:
    """Read triaged files into text, grouped by role (decoys are excluded by triage)."""
    out: dict[Role, str] = {}
    for role, paths in grouped.items():
        texts: list[str] = []
        for path in paths:
            if role in {Role.DB, Role.MEETING_NOTES, Role.REPORT_REQUEST, Role.GUIDANCE, Role.UNKNOWN}:
                texts.append(read_file(path))
            elif role is Role.IMAGE and path.name in image_text:
                texts.append(image_text[path.name])
        if texts:
            out[role] = "\n\n".join(texts)
    return out


def build_ledger(client_dir: Path, client, model, recorder: RunRecorder) -> ClientLedger:
    """Triage → OCR → extract (per source) → reconcile → ClientLedger."""
    with recorder.stage("triage"):
        grouped = triage_folder(client_dir)

    image_text: dict[str, str] = {}
    if grouped.get(Role.IMAGE):
        with recorder.stage("ocr"):
            for path in grouped[Role.IMAGE]:
                image_text[path.name] = read_image_text(path, recorder)

    sources = read_sources(grouped, image_text)
    db_text = sources.pop(Role.DB, "")
    accounts, _snapshot = parse_db(db_text) if db_text else ([], None)

    with recorder.stage("extract"):
        facts = extract_facts(client, model, accounts, sources, recorder)
    with recorder.stage("reconcile"):
        ledger = reconcile(accounts, facts)
    return ledger


def section_applies(section: dict, ledger: ClientLedger) -> bool:
    """Inclusion is deterministic: 'always' or a named ledger condition (e.g. 'disposal')."""
    rule = section.get("use_if", "always")
    if rule == "always":
        return True
    if rule == "disposal":
        return ledger.disposal
    if rule == "next_steps":
        # Only include a Next Steps section when there are loose ends to confirm.
        return bool(ledger.out_of_scope_accounts())
    return True


def generate_report(config: dict, ledger: ClientLedger, client, model, recorder: RunRecorder) -> str:
    instructions = config.get("global_instructions", "")
    sections = []
    for section in config["sections"]:
        if not section_applies(section, ledger):
            continue
        content = section["template"]
        for name, spec in section.get("placeholders", {}).items():
            value = fill_placeholder(name, spec, ledger, client, model, instructions, recorder)
            content = content.replace(f"<<{name}>>", value)
        sections.append({"title": section.get("title", ""), "content": content})
    return format_document(config, sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an advice report for a client.")
    parser.add_argument("--client", required=True, help="folder name under data/")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("config/template_config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--ledger-dir", type=Path, default=None, help="optional: also write the ledger JSON here")
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs"), help="where run telemetry is written")
    args = parser.parse_args()

    load_dotenv()
    client = build_client()
    model = model_name()
    recorder = RunRecorder(args.client, model)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    ledger = build_ledger(args.data_dir / args.client, client, model, recorder)
    with recorder.stage("generate"):
        report = generate_report(config, ledger, client, model, recorder)
    with recorder.stage("verify"):
        # A config can declare which checks apply (e.g. a non-advice doc has no Tax section).
        check_tax = bool(config.get("verification", {}).get("tax_section", True))
        problems = check_report(report, ledger, check_tax=check_tax)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A second document type generated from the same client gets a doc_id suffix so it doesn't
    # overwrite the advice report (e.g. client_02_medium__portfolio_review.md).
    doc_id = config.get("doc_id")
    out_name = f"{args.client}__{doc_id}" if doc_id else args.client
    out_path = args.output_dir / f"{out_name}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")

    if problems:
        print(f"  verification: {len(problems)} issue(s):")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("  verification: PASS")

    run_path = recorder.finalize(ledger, report, problems, args.runs_dir)
    print(f"  run log: {run_path} ({recorder.stages})")

    if args.ledger_dir:
        args.ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = args.ledger_dir / f"{args.client}.json"
        ledger_path.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {ledger_path}")


if __name__ == "__main__":
    main()
