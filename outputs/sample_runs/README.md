# Sample run telemetry

A curated snapshot — one real run per client — of the per-run telemetry the pipeline records. Each
JSON holds, for that generation:

- **every LLM call**: the prompt, the response, the latency, and which step it was (extraction per
  source, OCR, the generation prose slots);
- **per-stage timings** (`stage_seconds`) — handy for seeing where the time goes (extraction
  dominates) and that the independent calls run concurrently;
- the **model(s)** used (`gpt-4o` for extraction / `gpt-4o-mini` for generation);
- the reconciled **ledger** and the final **report**, for reference;
- the **verification** result.

The live telemetry directory (`outputs/runs/`) is gitignored — every run writes there. These four are
a deliberate sample so the prompts, outputs and timings can be inspected without re-running anything.
