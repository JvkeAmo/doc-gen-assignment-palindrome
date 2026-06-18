# Decisions

The hardest calls, why I made them (with the honest trade-off where there is one), and what I'd take
further. 

## A reconciled `ClientLedger` intermediate, not file-by-file fan-out

The real difficulty is reconciling conflicting sources, not formatting. So the pipeline is staged around one typed intermediate rather than fanning an agent out per file and merging summaries (which scatters conflicting facts across separate contexts):

```
triage → extract (per source, with provenance) → reconcile → generate off the ledger → verify
```

Extraction records what each source claims (it does not resolve conflicts). Reconciliation is the one place conflicts are resolved, producing a single `ClientLedger` plus an auditable conflict log.
Generation reads only a clean slice of the ledger, so files which may not be necessary and raw conflicts can never reach the report. Verification asserts the output against the ledger. Each stage is independently testable (as this is the best way to evaluate a framework depending on agents), and "which source do we trust" lives in one auditable place.

**Why it's done this way:** the brief grades "which source do you trust when sources disagree" above almost everything else, so the whole point is to make that decision in one auditable place rather than scatter it across separate per-file agents. Staging it this way also makes each step independently testable — the only real way to trust a system built on LLM calls.

## OpenAI, with a stronger model for extraction than generation

I selected `EXTRACT_MODEL=gpt-4o` for extraction; `LLM_MODEL=gpt-4o-mini` for generation and the statement-image vision call. Extraction precision (scope, figures) is where mistakes are expensive, and `gpt-4o-mini` is good enough for the generation of the summary. So: precision where it matters, cheap prose elsewhere, lowering costs overall, while maintaing a top tier product. Realistically, due to pricing we could have also gone for a variant of the GPT-5 family (pricing is similar). 

## Extraction via structured outputs, one schema per source

Each prose source is read with OpenAI structured outputs (`client.chat.completions.parse`) against its OWN Pydantic schema (`RequestFacts`, `MeetingFacts`, `StatementFacts`, `ExploreFacts`); and the partials are then merged. 

The db itself is parsed deterministically (joint accounts de-duped). The calls are independent and run concurrently (speed up at inference). Crucially, structured outputs guarantee the *shape*, not the *semantics* so a wrong-but-well-typed scope still passes (this is not verification of outputs, just structure), which is why verification + the golden ledgers remain the real correctness check (and why the extraction model choice mattered).

**Why it's done this way:**
Per-source schemas rather than one big call allow us to:
- **Focus** — a call can only return fields relevant to its source, so fields can't bleed across calls,
  and the per-field *descriptions* carry the extraction rules (scope = covered-only; live values must be stated, never computed; exclude non-actioned aspirations). The schema is the contract, so the prose prompts stay tiny, verifiable and easily updatable in case we encounter drift/need to tweak both inputs or outputs.
- **Robustness** — the parsed result is shape-guaranteed, allowing post-processing of the outputs.

## Prefer Deterministic extraction in cases where it's available. Use LLM for decisions that require reasoning

Numbers, dates, dedupe, section inclusion and the funds arithmetic are deterministic; the LLM only reads prose. The clearest example is `funds.py`: `available_to_invest = sum(available inflows) − sum(committed outflows)`, contingent money excluded.
The LLM does the fuzzy part (classifying each external fund `available`/`contingent`/`committed`); the tool does the arithmetic the model can't be trusted with. The result is written into the ledger as a sourced figure, so the recommendation can state it and verification accepts it (client_04: £850k − £200k bridging = £650k investable, the £400k earnout excluded).

## Generation off the ledger, with a bounded reflection loop

Once the facts are reconciled into the ledger, the report is built section by section from those facts not from the original files. Most sections are produced by plain code with no AI involved (the accounts table, the fees section, the opening scope sentence, the tax statement), so their numbers are exact. Only two sections are actually written by the model: the short high-level summary and the recommendation. The legally-required lines like the FCA authorisation line and the risk warning are *fixed text we drop in word-for-word*, so the model never writes those.

After the model writes one of its two sections, code checks it against the ledger. If the model has slipped in a figure that isn't in the ledger, or a number it worked out itself (for example, writing "each ISA receives £22,500" a split it calculated rather than a figure we gave it), the check rejects it and asks the model to try again, telling it exactly what was wrong. It allows up to two retries. If the section still isn't right, we ship it with a visible **[FLAG: needs review]** note rather than quietly letting the bad version through.

**Why it's done this way:** in financial advice a made-up or miscalculated number must never reach the client, so the model is only ever allowed to *phrase* facts we've already verified — never to invent or calculate figures. And when a slip can't be fixed automatically we'd rather show a visible flag than let a confident-looking wrong number through — the right level of control for a regulated document.

## Two ways of checking "correct": rules on the report, and hand-written expected answers

We're given no official right answers, so part of the task is deciding what "correct" means and proving it. I check it two ways (both in `evaluate.py`), because it means two different things here.

The first is a set of **rules applied to the finished report** that must hold for any client: the required legal lines are present word-for-word, the Tax section appears only when something is actually being sold, anything we're waiting on a human for is flagged rather than guessed, the accounts table matches the ledger, and — the important one — every number in the report traces back to a number in the ledger. That last rule quietly does a lot of work: a decoy figure from a distractor file, or a number the model invented, won't be in the ledger, so it's caught automatically (for example, if the report said "we project £99,000 next year" and £99,000 isn't a reconciled fact, the check flags it).

The second is a set of **expected answers I wrote by hand for each example client** — what the reconciled facts should come out as: the right values and dates, which accounts are in scope, which source won each conflict, the funds total. Comparing the pipeline's ledger against these checks the deterministic heart — the extraction and reconciliation — directly (for example, the answer for client_04 lists exactly 7 in-scope accounts, so if extraction wrongly drags in a cash account, this fails immediately). It only checks the stable factual fields, not the exact wording, which can reasonably vary between runs.

These hand-written answers live only in the test folder, never in the product, so they are not the kind of overfitting the brief warns about — baking an example client's specifics into the live pipeline. I actually went the other way and removed an earlier shortcut that hardcoded the four examples' decoy numbers to block them: that one was overfit (it would do nothing on a different client), and the general "every number must trace to the ledger" rule already covers it properly.

**Why it's done this way:** the two checks measure different things on purpose. The rules confirm the *output* is safe and well-formed; the expected answers confirm the *reasoning underneath* — which source we trusted, what's in scope — is actually right, which is the part the brief weighs most heavily. Separating them means a failure tells you *where* it broke: a bad final report versus a bad reconciliation. And building the rules as general principles, while keeping the example answers as throwaway test fixtures, is what lets the same checks work on the held-out clients we'll never see. It paid off concretely — writing and double-checking those expected answers is what surfaced a real scope mistake that would otherwise have slipped through.

## A third, softer check: an AI judge — but only offline, never while generating

The two checks above can only test hard facts. They can't tell whether the writing is actually any good — whether it reads clearly, stays high-level where it's supposed to, and handles sensitive situations tactfully. So there's a third, separate check: a second AI reads the finished report and scores it one-to-five on those softer qualities (for example, it gave client_03 5/5 on sensitivity for acknowledging that Jean's mother had recently died).

The important part is that this judge only ever runs as an offline evaluation tool — it is never in the loop while a real client's report is being generated.

**Why it's done this way:** a model judging quality is slower and a bit inconsistent from run to run, so it isn't something to trust in the live path, where one flaky score could wave through a bad change or block a good one. It belongs offline, where a person reads the scores and the run-to-run noise averages out. The thing that actually guards generation is the fast, strict figure-checker from the section above; the judge measures something different — quality rather than rule-following — so the two complement each other instead of overlapping.

## A second kind of document is just a new config file, not new code

Because every section is built from the same reconciled facts, and all the wording lives in the config, producing a completely different *type* of document needs no new code — just a new config file. The example I included is an internal "review sheet" for the adviser rather than the client: it lays out which source won each conflict and why, where every value came from, and every outstanding flag (for example, it spells out "chose £38,000, observed live in the meeting, over £30,000 from the older database snapshot — most-recent wins"). It's marked as internal, so it skips the client-facing rules — it deliberately shows the rejected and stale numbers, and the closed accounts, that a real client report would never include.

**Why it's done this way:** it proves the central design claim — that the *data* (the reconciled facts) and the *presentation* (the document) are genuinely separate, which is one of the brief's optional asks (support more than one document type without duplicating everything). And making that second document the review sheet does double duty: it turns "which source did you trust when they disagreed" — the thing the brief grades hardest — into something you can actually read, instead of leaving it buried in the raw data.

## Handling a file we've never seen: read it separately, use it, but flag it

We work out what each file is from its name. That's fine for the known files, but a held-out client could include something we've never seen. Two tempting options are both bad: feed it through the normal extraction (which only looks for specific known fields, so anything genuinely new is silently ignored), or just ignore the file (and lose real information). Instead, an unrecognised file gets its own separate, more open-ended read — and that read only happens when such a file actually exists, so it costs nothing the rest of the time. It can pull out values for accounts we already know, spot an entire account that isn't in the database, and list anything it found but couldn't categorise (for example, a stray "trust deed" note reveals an offshore bond we have no record of — it gets added to the report, but labelled "found in an unrecognised document, please verify"). Whatever it finds is used but always flagged for a human, never silently trusted; anything it couldn't place is surfaced as a review note rather than thrown away.

**Why it's done this way:** the brief says the real test is a held-out set with the same themes but different specifics, so the system has to degrade gracefully on something it wasn't built for. The guiding rule is "route, don't drop": never silently trust a source we haven't vetted, and never silently discard material a person might need. It does nothing on the four example clients (none of them has an unknown file) and is covered by tests — it's insurance for the held-out run, not something tuned to these examples.

## Concurrency: overlap every independent call

The independent model calls (per-source extraction, OCR, the prose slots, the eval judge) run concurrently, with ordering preserved where it matters so output stays deterministic. On OpenAI this cuts the dominant extract stage and the generation stage to roughly their slowest single call.

## What I'd take further
- Use the judge to drive prompt tuning — a judge-scored A/B over prompt variants, committing the winner.
- Per-value "unknown" provenance for a *known* account's value taken from an unrecognised source.
- A direct vision pass (read + interpret in one step) for unrecognised images, vs. the current OCR→text.
