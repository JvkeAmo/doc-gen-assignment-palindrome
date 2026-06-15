# Decisions

A running log of the hardest calls and why, and what I would take further with more time.

## Architecture: a reconciled `ClientLedger` intermediate (not file-by-file fan-out)

The real difficulty here is **reconciling conflicting sources**, not formatting. So rather than
fan an agent out per file and merge summaries (which scatters conflicting facts across separate
contexts), the pipeline is staged around one typed intermediate:

```
triage → extract (with provenance) → reconcile → generate off the ledger → verify
```

- **Extraction** turns each source into tagged claims (value + source + date); it does **not**
  resolve conflicts.
- **Reconciliation** is the one place conflicts are resolved, producing a single `ClientLedger`
  plus an auditable conflict log.
- **Generation** reads only a clean slice of the ledger, so decoy figures and raw conflicts can
  never reach the report.
- **Verification** asserts the output against the ledger.

Why: separates parse / decide / write into independently testable stages, and puts "which source
do we trust" in one auditable place — the thing the brief explicitly grades.

## LLM provider: local Ollama by default, OpenAI-swappable

Development and testing run against a **local Ollama** server (`qwen3:8b`) via its OpenAI-compatible
API, so iteration costs no hosted credits. Provider is env-driven (`LLM_BASE_URL` / `LLM_API_KEY` /
`LLM_MODEL`), so a clean checkout can point at OpenAI for the final run without code changes. A small
`llm.py` wraps client creation and strips qwen3's `<think>` preamble.

> Note for re-running from a clean checkout: the default `.env.example` targets Ollama. Set the
> `LLM_*` OpenAI values (or run Ollama with `qwen3:8b` pulled) before generating.

## Baseline kept on record

Ran the unmodified starter on `qwen3:8b` to capture a "before" (`outputs/client_01_clean.md` at the
baseline commit). It exhibits the failure modes the redesign targets: the verbatim FCA line gets
paraphrased and duplicated, the Background leaks transaction amounts and out-of-scope intentions,
two inconsistent holdings tables appear, and the conclusion regenerates a whole mini-report. The
section-inclusion gate did correctly omit Tax (no disposal).

## What the first end-to-end build does

- **Triage** routes files by role and drops decoys (market update, portfolio pack) before they can
  reach a prompt.
- **Extraction** parses the db deterministically (de-duping joint accounts) and reads the prose
  sources + OCR'd images into one typed `ExtractedFacts` via a single LLM call.
- **Reconciliation** applies house rules — recency-wins (with a conflict log), null→flag, finalise
  figures (CGT, fee rates) → flags, scope, disposal — into one `ClientLedger`.
- **Generation** fills each section from a ledger slice: deterministic renderers for the holdings
  table, fees, scope and the CGT statement; LLM prose only for the summary and recommendation;
  verbatim lines (FCA, risk warning) are static template text.
- **Verification** checks the report against the ledger and runs after every generation; the same
  checks back an `evaluate` harness and a pytest suite.

All four clients run end to end on local `qwen3:8b`. The hardest traps are handled: stale-vs-fresh
values (incl. client_04's image agreeing with the stale db), joint de-dup, closed/null accounts,
the contingent earnout, and the conditional Tax section.

## Known limitation / next iteration

On the small local model, prose generation occasionally under-uses the ledger (e.g. client_02's
recommendation quoted the stale £40k and a £20k split rather than the reconciled £45k). The ledger
is correct; the weak link is the 8b model's prose fidelity. The verifier *catches* this. Next steps:
a stronger generation model, stricter prompts, and/or a critic→revise loop; netting committed money
(client_04's £200k bridging) numerically; per-client guidance currently over-captures process notes.

## To take further (noted, not yet done)
- Reconciliation as "LLM proposes, code checks": deterministic assertions over every mechanical
  decision (recency-wins, dedupe, null→flag, scope match).
- Verification doubling as both an offline eval and a pre-send guard.
- A reflection (critic→revise) loop and prompt auto-tuning against the eval.
