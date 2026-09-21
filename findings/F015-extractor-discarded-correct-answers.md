---
id: F015
title: The answer extractor silently discarded 58.6% of correct answers
kind: retraction
status: retracted
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T04-32Z_h1, 2026-09-21T07-41Z_backend]
  analysis: [analysis/out/tables/quality.md]
  code_sha: 2e2dc98
supersedes: [F014]
---

## Claim (withdrawn)

Every task-accuracy number this study has produced is wrong, and wrong in one
direction. `extract_gsm8k_answer` returned "no answer" for **150 of 256**
(58.6%) of Qwen3-4B's GSM8K generations that in fact contained a correct,
plainly stated answer. After the fix the same 256 generations yield **0**
unparseable.

## The defect

Three faults compounding, all in one function:

1. The number pattern `-?[\d,]+\.?\d*` requires no digit, so it matches a
   **bare comma**.
2. The answer-statement pattern had no word boundary on its keywords, so
   `so` matched inside the ordinary sentence opener `"So,"`.
3. On a parse failure that stage returned `None` **immediately** instead of
   falling through to the remaining strategies.

Together: text containing `"So,"` matched stage 2, captured `","`, `float("")`
raised, and the function returned `None` — abandoning the search before the
last-number fallback could find the answer sitting a few characters away.

A representative victim ends `**Answer: 3 bolts**.` with reference answer 3.

## Why it was not caught earlier

It produces a plausible number, not an error. Low accuracy on a small model
with thinking disabled looked like a believable result, and it was
corroborated by two *other* genuine defects found in the same area — wrong
reference answers (F013) and truncation (F014) — each of which supplied a
satisfying explanation for the low numbers and made the extractor look
innocent.

**This supersedes F014's conclusion.** Raising the token budget fixed
truncation (69 capped generations to 1) but barely moved the unparseable count
(117 to 121), because truncation was never its main cause. F014's *fix* stands
— 256 tokens genuinely truncated a quarter of generations — but its
attribution of `unparseable` to truncation was wrong.

## What caught it

Insisting on an explanation for `unparseable` staying flat after the budget
rose. The number that should have fallen and did not was the only signal; had
the budget change moved it even a little, the extractor would still be broken.

## Consequence

No GPU time is wasted. The generations were always valid — only the scoring
was wrong — so `quality_compare.load_quality` now re-scores from the stored
text and the analysis layer treats the `correct` field written during a run as
advisory. Scoring is a pure function of recorded output and must never require
re-running hardware to improve.

## What this does NOT establish

The corrected accuracies are not yet reported: this was diagnosed while the
quality sweep was still in flight. No claim about FP8 and answer quality is
supportable until those land and the paired comparison runs.

## How to reproduce

```bash
python analysis/build_tables.py   # re-scores; see analysis/out/tables/quality.md
```
