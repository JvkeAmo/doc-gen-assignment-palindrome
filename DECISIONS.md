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
- **Extraction** parses the db deterministically (de-duping joint accounts) and reads each prose
  source via its own focused LLM call (see "Per-source extraction" below).
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

## Per-source extraction (not one big call)

Each free-text source gets its own focused LLM call — the report request, the meeting note (+ internal
notes), and any statement image — each anchored by the db account digest so entity-matching survives,
then the partial `ExtractedFacts` are merged (first-non-null for scalars, union for lists).

Why: a small model drops fewer fields on a tight, single-purpose JSON than on a sprawling 9-key one,
and the calls are independent (parallelisable later). The shared db anchor preserves cross-source
knowledge, and the expensive cross-referencing (recency, dedupe) happens deterministically in
reconcile anyway. This split immediately surfaced two bugs the single call had hidden: a source
returning `guidance` as a string (now coerced scalar→list) silently emptied a whole extraction, and
`investment_amounts` was inferring the stale db value (now restricted to figures written explicitly in
the document). Trade-off: ~3 sequential calls instead of 1 (extraction ~75–95s on `qwen3:8b`).

## Why the work sits in `src/`, not only the config

The brief says "most of your work goes in the config" — that describes the *starter's* design, where
the pipeline is a dumb loop and prompts are the only lever. It also says "improve the pipeline; it's
just llm calls". We deliberately moved logic (extraction, reconciliation, verification) into `src/`,
because compliance-critical behaviour — verbatim text, "never invent a CGT figure", section inclusion —
should be enforced by code, not left to a prompt's goodwill. The config still owns the report
definition (sections, inclusion rules, the LLM prompts, verbatim text); the few prompts that remain are
the ones that genuinely need a model, which is what makes iterating/tuning them tractable.

## Run telemetry

Every run writes `outputs/runs/<client>_<ts>.json` (per-stage timings, every prompt + response +
latency, the ledger, the report) and appends to `outputs/runs/index.md`, for inspecting prompts,
outputs and latency. Gitignored (local dev telemetry, not part of the deliverable).

## Reflection loop (generate → critique → revise)

LLM prose slots are now generated through a bounded critique→revise loop: generate, run a
slot-scoped critic (`verify.critique_slot` — summary must carry no monetary figures; recommendation
may only state ledger-sourced figures, no computed splits), and on failure re-prompt with the
specific problems (max 2 retries; at temperature 0 the prompt must change each time, which the
feedback does). If still failing, the section ships with a visible `[FLAG: needs review]` rather than
hiding the issue. The critic and the final verifier share `money_tokens`/`allowed_figures`, so they
agree on what counts as a figure — including £/$/€ and bare comma-grouped numbers (a real hole: the
model sometimes writes `$22,500`, which a £-only check missed). The critic guarantees figures are
*sourced*, not *semantically apt* — judging aptness/tone is left to a future LLM-judge.

## Evaluation: two layers, plus a removed overfit

"Correct" is defined two ways, both in the eval harness (`evaluate.py`):
- **Report rules** (`verify.check_report`) — invariants on the final report (verbatim lines, Tax iff
  disposal, gaps flagged, every figure sourced, table consistent).
- **Golden ledgers** (`eval/golden/*.json`, `golden.compare_ledger`) — hand-authored expected
  reconciled facts for each example (values, dates, scope, disposal, conflicts, funds, gaps),
  evaluating extraction+reconciliation directly. Stable fields only; free-text is not asserted. These
  are test fixtures, not production logic — not the overfitting the brief warns against.

Removed `DECOY_FIGURES`: it hard-coded the four examples' specific decoy numbers (overfit — the
held-out set differs) and was redundant with the general "every figure must trace to the ledger"
check. Also made figure detection currency-agnostic (£/$/€) and catch bare comma-grouped numbers.

## LLM-judge — eval only, not inference

`judge.py` scores prose quality (faithfulness, background altitude, sensitivity, clarity) against the
ledger — the dimensions deterministic checks can't measure (e.g. whether the inheritance was handled
tactfully). Run via `evaluate --judge`. It is deliberately **not** used at inference: the runtime
guard is the fast, deterministic `critique_slot`; the judge is slower/flakier and belongs where a
human reads the scores and noise averages out. The two are complementary (compliance vs quality), not
redundant.

## Funds calculator: LLM classifies, code computes

`funds.py` computes `available_to_invest = sum(available inflows) − sum(committed outflows)`, with
contingent money (an unreceived earnout) excluded. The LLM does the fuzzy part — classifying each
external fund as `available` / `contingent` / `committed` during extraction; the deterministic tool
does the arithmetic the model can't be trusted with. The result is written into the ledger as a
sourced figure, so the recommendation can state it and verification accepts it (an LLM-computed number
would be unsourced and often wrong). client_04: completion £850k − bridging £200k = £650k investable,
earnout £400k excluded. Asserted by the golden ledgers.

## To take further (noted, not yet done)
- Use judge scores to drive prompt tuning (the "prompts as code" loop).
- An exploratory pass for unrecognised documents (discovery + review flags).
- Multiple document types reusing the ledger (config-only).
