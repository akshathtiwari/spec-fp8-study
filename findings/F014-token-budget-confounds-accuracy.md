---
id: F014
title: A 256-token budget truncates GSM8K reasoning and is scored as no answer
kind: method
status: established
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T04-32Z_h1, 2026-09-21T07-41Z_backend]
  analysis: []
  code_sha: 86e97b0
---

## Claim

At `max_tokens: 256`, **27%** of Qwen3-4B's GSM8K generations hit the cap
mid-reasoning (69 of 256 examined, across all Qwen3-4B cells). On the harder
problems in the 256-problem quality set the rate is higher: the first quality
cell reported 117 of 256 unparseable (46%). A truncated generation yields no
extractable answer, so at that budget the measurement reports **how often the
model runs out of tokens**, not how often it is right.

## Evidence

Examined every GSM8K generation (prompt indices 16-31) stored in
`results/requests/` for Qwen3-4B cells:

| signal | count | share |
|---|---|---|
| hit the 256-token cap | 69 / 256 | 27.0% |
| contain `<think>` | 0 / 256 | 0.0% |

Truncated outputs are ordinary verbose reasoning cut mid-sentence, e.g. ending
`### Step 4: Calculate Profit\nProfit is the selling price minus the total
cost.\n$$\n` — the model was still working.

## Reasoning

Two candidate causes were distinguishable and only one survives. Thinking mode
is **not** the problem: zero generations contain `<think>`, so the
non-thinking chat template is being applied correctly. The cause is purely the
token budget, which was inherited from the gate where 256 was chosen to keep
the per-cell cost low.

This matters beyond precision. If a precision change alters verbosity even
slightly, it shifts the truncation rate, and that shift lands directly on the
accuracy comparison it is supposed to inform — a confound pointing straight at
the question being asked.

The gate's 27% understates the effect for the quality set because the gate uses
GSM8K test indices 0-15, which are among the easiest problems; the quality set
spans 256 problems including substantially harder ones.

## What this does NOT establish

That 768 tokens is sufficient — only that 256 is not. `truncated` is now
reported alongside `unparseable` in every quality record precisely so the
budget's adequacy is visible rather than assumed. A high rate at 768 means the
budget must rise again; it is a measurement artifact and must never be read as
a result.

## Fix

The quality stage sets its own `max_tokens` (768) rather than inheriting the
gate's, and reports `truncated` per cell. Concurrency raised 16 -> 32 to
absorb the longer generations, which is safe because task accuracy, unlike
exact match, does not need concurrency 1 (F009). The probe function timeout
rises to 120 minutes; resume and per-60s volume commits cap the cost of
hitting it at one re-run cell.

## Cost note

The first quality sweep was stopped after one cell rather than run to
completion, once `unparseable=117` made clear the measurement was
truncation-dominated. Finishing it would have spent roughly 50 more GPU-minutes
producing eight numbers that could not answer the question.
