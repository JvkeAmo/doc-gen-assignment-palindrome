# doc-gen-assignment

A config-driven pipeline that turns a client's mixed source material (structured db, meeting
notes, an instruction sheet, statement images) into a finished investment advice report.

The real work here is **reconciling conflicting sources**, not formatting. So the pipeline is built
around one typed intermediate — a `ClientLedger` — that every report section is generated from. See
`DECISIONS.md` for the reasoning and `PROJECT_GUIDANCE.md` for the original brief.

## Setup

```bash
# clone your fork, then:
cp .env.example .env          # defaults to a local Ollama server (no API credits needed)
uv sync                       # add --extra dev for the tests
```

By default the pipeline uses a **local Ollama** model (`qwen3:8b`) via its OpenAI-compatible API,
so iteration costs nothing. Pull the models and run Ollama first:

```bash
ollama pull qwen3:8b
ollama pull deepseek-ocr      # used to read statement images
```

To use hosted OpenAI instead, set the `LLM_*` variables in `.env` (see `.env.example`).

## Run

```bash
uv run python -m agent_pipeline.generate --client client_01_clean --ledger-dir outputs/ledgers
# report  -> outputs/client_01_clean.md
# ledger  -> outputs/ledgers/client_01_clean.json   (the reconciled facts, for inspection)
```

Clients live under `data/`: `client_01_clean`, `client_02_medium`, `client_03_hard`,
`client_04_stretch` (large and messy on purpose).

## Evaluate

We are given no expected outputs, so "correct" is defined by deterministic checks (verbatim lines
present, Tax section iff a disposal, no decoy or unsourced figures, human-finalise gaps shown as
flags, holdings table consistent with the ledger). They run automatically after each generation and
can be run standalone over the generated artifacts:

```bash
uv run python -m agent_pipeline.evaluate     # checks outputs/ against outputs/ledgers/
uv run pytest                                 # deterministic unit tests
```

## How it fits together

```
data/<client>/*                      db (json), meeting notes, report request, images, fde notes
        │
        ▼
triage.py        classify each file by role; drop decoys (general market/portfolio docs)
        │
        ▼
extract.py       db parsed deterministically; prose + OCR'd images -> facts via one LLM call
        │
        ▼
reconcile.py     merge into ONE ClientLedger: dedupe joint accounts, recency-wins values
        │         (+ conflict log), null -> flag, finalise figures -> flags, scope, disposal
        ▼
render.py        each section generated from a ledger slice: deterministic renderers for tables/
        │         fees/scope; LLM prose for summary/recommendation; verbatim lines are static
        ▼
verify.py        deterministic checks of the report against the ledger
        │
        ▼
outputs/<client>.md   (+ outputs/ledgers/<client>.json)
```

`src/document_formatter/formatting.py` (final assembly) is unchanged from the starter, as advised.
