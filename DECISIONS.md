# Decisions

The hardest calls, why I made them (with the honest trade-off where there is one), and what I'd take
further. This reflects the **final** design, not the order it was built in.

## A reconciled `ClientLedger` intermediate, not file-by-file fan-out

The real difficulty is **reconciling conflicting sources**, not formatting. So the pipeline is staged
around one typed intermediate rather than fanning an agent out per file and merging summaries (which
scatters conflicting facts across separate contexts):

```
triage → extract (per source, with provenance) → reconcile → generate off the ledger → verify
```

Extraction records what each source claims (it does not resolve conflicts). **Reconciliation is the
one place conflicts are resolved**, producing a single `ClientLedger` plus an auditable conflict log.
Generation reads only a clean slice of the ledger, so decoys and raw conflicts can never reach the
report. Verification asserts the output against the ledger. Each stage is independently testable, and
"which source do we trust" lives in one auditable place — the thing the brief explicitly grades.

## OpenAI, with a stronger model for extraction than generation

`EXTRACT_MODEL=gpt-4o` for extraction; `LLM_MODEL=gpt-4o-mini` for generation and the statement-image
vision call. Extraction precision (scope, figures) is where mistakes are expensive, and `gpt-4o-mini`
proved unreliable on the scope-exclusion judgment — it pulled uncovered cash accounts into scope and
no prompt/schema wording fixed it, whereas `gpt-4o` gets it right. Generation prose is easy enough for
the mini model. So: precision where it matters, cheap prose elsewhere — a few cents per run. Models and
endpoint are env-overridable; the key resolves from `OPENAI_API_KEY` / `OPENAI_KEY` / `LLM_API_KEY`.

## Extraction via structured outputs, one schema per source

Each prose source is read with OpenAI **structured outputs** (`client.chat.completions.parse`) against
its OWN Pydantic schema (`RequestFacts`, `MeetingFacts`, `StatementFacts`, `ExploreFacts`); the partials
are then merged. Why per-source schemas rather than one big call:
- **Focus** — a call can only return fields relevant to its source, so fields can't bleed across calls,
  and the per-field *descriptions* carry the extraction rules (scope = covered-only; live values must be
  stated, never computed; exclude non-actioned aspirations). The schema is the contract, so the prose
  prompts stay tiny.
- **Robustness** — the parsed result is shape-guaranteed, so there is no tolerant JSON parsing or
  scalar/dict coercion to maintain.

The db itself is parsed deterministically (joint accounts de-duped). The calls are independent and run
concurrently. Honest boundary: structured outputs guarantee the *shape*, not the *semantics* — a
wrong-but-well-typed scope still passes, which is why verification + the golden ledgers remain the real
correctness check (and why the extraction model choice mattered).

## Logic in `src/`; the config owns the wording

The brief says "most of your work goes in the config", but that describes the *starter* — a dumb loop
where prompts are the only lever; it also says "improve the pipeline". Compliance-critical behaviour
(verbatim text, "never invent a CGT figure", section inclusion, recency-wins, dedupe) is **enforced by
code**, not left to a prompt's goodwill. But every **word the client reads is config-driven**: each
`render:` placeholder carries a `template` in `template_config.json` and the renderer only supplies
values/flags into it. So `config/` owns the report definition and all wording; `src/` owns the logic;
the holdings table stays code-built (structural, not prose). Trade-off: a reviewer expecting a fatter
config might read this as not following the letter of the brief — mitigated by its being the right call
for reliability, and by this log.

## Deterministic where it counts, LLM only where needed — the funds calculator

"LLM proposes; code checks and computes." Numbers, dates, dedupe, section inclusion and the funds
arithmetic are deterministic; the LLM only reads prose. The clearest example is `funds.py`:
`available_to_invest = sum(available inflows) − sum(committed outflows)`, contingent money excluded.
The LLM does the fuzzy part (classifying each external fund `available`/`contingent`/`committed`); the
tool does the arithmetic the model can't be trusted with. The result is written into the ledger as a
**sourced** figure, so the recommendation can state it and verification accepts it (client_04:
£850k − £200k bridging = £650k investable, the £400k earnout excluded).

## Generation off the ledger, with a bounded reflection loop; verbatim is static

Each section is filled from a ledger slice: deterministic renderers (table, fees, scope, CGT) and LLM
prose only for the summary and recommendation. Verbatim lines (FCA, risk warning) are **static template
text**, never generated. Prose slots run through a bounded **critique → revise** loop: a slot-scoped
critic (`verify.critique_slot`) rejects unsourced or computed figures, the prompt is re-issued with the
specific problems (max 2 retries), and if it still fails the section ships with a visible
`[FLAG: needs review]` rather than hiding the issue. This is the "agentic" part — bounded, verify-driven
self-correction rather than open-ended orchestration, which is the right shape for a regulated document.

## Evaluation: report rules + golden ledgers, and no overfitting

"Correct" is defined two ways in `evaluate.py`:
- **Report rules** (`verify.check_report`) — verbatim lines, Tax iff disposal, gaps flagged, table
  consistent, and **every figure traces to the ledger** (this generally subsumes decoy detection).
- **Golden ledgers** (`eval/golden/*.json`) — hand-authored expected reconciled facts (values, dates,
  scope, disposal, conflicts, funds, gaps): a precise check on the deterministic heart, lenient on
  free-text.

The goldens are test fixtures in `eval/`, not production logic, so they are **not** the overfitting the
brief warns against — and I removed an earlier `DECOY_FIGURES` denylist that *was* overfit (it baked the
four examples' decoy numbers; the general figure-sourcing check covers it). The goldens earned their
keep: authoring/auditing them caught a real scope error that a held-out run would otherwise have failed.

## LLM-judge: eval-only, not in the inference loop

`judge.py` scores prose *quality* (faithfulness, background altitude, sensitivity, clarity) — the
dimensions deterministic checks can't measure. It is deliberately **not** used at inference: the runtime
guard is the fast, deterministic critic; the judge is slower and flakier and belongs offline where a
human reads scores and noise averages out. Complementary (compliance vs quality), not redundant.

## A second document type is just another config

Because every section generates from the same ledger and all wording lives in config, a second document
type needs **no new `src/` logic**. The included example (`adviser_review_config.json`) is an internal
**reconciliation review sheet**: the conflict log (which source we trusted, and why), per-account
provenance, and all flags — i.e. it demonstrates "which source to trust" as a document, not buried in
JSON. It sets `"client_facing": false`, so it gets a minimal render check instead of the client-report
rules (an audit view legitimately shows stale/rejected values and closed accounts).

## Unknown documents: route, explore, flag — never silently miss

Triage is filename-based, so an unfamiliar file is a real risk. Instead of folding it into the
structured extraction (where novel content is silently skipped) or dropping it, an unrecognised file
goes to `Role.UNKNOWN` and a separate **exploratory pass that fires only when one is present**. It can
discover accounts not in the db and list `unmapped` material; reconcile *uses but flags* it (provenance
`unknown`, review flags surfaced under the table). Route-don't-drop: never silently trust an unvetted
source, never silently discard material. A held-out robustness mechanism; idle on the four examples,
proven by tests.

## Concurrency: overlap every independent call

The independent model calls — per-source extraction, OCR, the prose slots, the eval judge — run
concurrently (thread pools; the work is I/O-bound), with ordering preserved where it matters so output
stays deterministic. On OpenAI this cuts the dominant extract stage and the generation stage to roughly
their slowest single call.

## What I'd take further
- Use the judge to drive prompt tuning — a judge-scored A/B over prompt variants, committing the winner
  (prompts-as-code via visible history).
- Per-value "unknown" provenance for a *known* account's value taken from an unrecognised source.
- A direct vision pass (read + interpret in one step) for unrecognised images, vs. the current OCR→text.
