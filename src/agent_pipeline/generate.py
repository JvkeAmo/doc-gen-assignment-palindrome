"""Generate an advice report for a client from the template config.

Usage:
    python -m agent_pipeline.generate --client client_01_clean
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent_pipeline.llm import build_client, model_name, strip_thinking
from document_formatter.formatting import format_document
from document_formatter.loading import read_file


class ReportGenerator:
    """Builds a report one section at a time from the template config."""

    def __init__(self, openai_client: OpenAI, model: str) -> None:
        self._openai = openai_client
        self._model = model

    def generate(self, config: dict, context: str) -> str:
        instructions = config.get("global_instructions", "")
        sections = []
        for section in config["sections"]:
            if not self._section_applies(section, context, instructions):
                continue
            sections.append(
                {
                    "title": section.get("title", ""),
                    "content": self._build_section(section, context, instructions),
                }
            )
        return format_document(config, sections)

    def _section_applies(self, section: dict, context: str, instructions: str) -> bool:
        rule = section.get("use_if", "always")
        if rule == "always":
            return True
        verdict = self._ask(
            f"{instructions}\n\n"
            f"Decide whether this section applies to the client.\n"
            f"Rule: {rule}\n"
            f"Reply with only 'yes' or 'no'.",
            context,
        )
        return verdict.lower().startswith("y")

    def _build_section(self, section: dict, context: str, instructions: str) -> str:
        content = section["template"]
        for name, spec in section.get("placeholders", {}).items():
            value = self._ask(f"{instructions}\n\n{spec['prompt']}", context)
            content = content.replace(f"<<{name}>>", value)
        return content

    def _ask(self, instruction: str, context: str) -> str:
        response = self._openai.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "user", "content": f"{context}\n\n---\n\n{instruction}"}
            ],
        )
        return strip_thinking(response.choices[0].message.content)


def read_client_context(client_dir: Path, filenames: list[str]) -> str:
    """Read the named files from the client folder and concatenate them into one context string."""
    parts = []
    for name in filenames:
        parts.append(f"=== {name} ===\n{read_file(client_dir / name)}")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an advice report for a client."
    )
    parser.add_argument("--client", required=True, help="folder name under data/")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--config", type=Path, default=Path("config/template_config.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    load_dotenv()
    generator = ReportGenerator(build_client(), model_name())

    config = json.loads(args.config.read_text(encoding="utf-8"))
    client_dir = args.data_dir / args.client
    filenames = sorted(path.name for path in client_dir.iterdir() if path.is_file())
    context = read_client_context(client_dir, filenames)
    report = generator.generate(config, context)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.client}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
