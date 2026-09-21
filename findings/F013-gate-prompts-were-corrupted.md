---
id: F013
title: The correctness gate's GSM8K reference answers were partly wrong
kind: retraction
status: retracted
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T04-32Z_h1, 2026-09-21T07-41Z_backend]
  analysis: [analysis/out/tables/compatibility_matrix.md]
  code_sha: dea69b2
supersedes: []
superseded_by: []
---

## Claim (withdrawn)

Every `task_accuracy` number produced before 2026-09-21 was scored against a
prompt set whose GSM8K half was **not GSM8K**. The 16 problems were paraphrases
of real GSM8K items that retained the *originals'* reference answers, so at
least two answers were arithmetically inconsistent with their own prompt text.
All previously reported accuracies are therefore understated by an unknown
amount and must not be cited.

## Evidence

Checked against `openai/gsm8k`, config `main`, split `test` (all 1319
problems), matching on whitespace- and quote-normalised text.

Only 10 of 16 appeared verbatim. Of the 6 that did not:

| # | divergence | gate answer | correct for the gate's own text |
|---|---|---|---|
| 0 | "sells **every duck egg**" vs original "sells **the remainder**" | 18 | **32** (16 x $2) |
| 13 | asks "how many **minutes**" vs original "what **speed**" | 0 | **60** (12 mi @ 4 mph = 3 h; 2 h used; 6 mi in 1 h) |
| 14 | merchant/jewelry problem absent from GSM8K entirely | -4600 | unverifiable |
| 5, 8 | paraphrased, but answer updated consistently | 9, 120 | self-consistent |
| 7 | original contains a typo ("how load"); gate fixed it | 160 | unchanged |

A negative reference answer (-4600) is on its own a strong signal, since GSM8K
answers are essentially always positive quantities.

## Consequence

A model answering perfectly could score at most 14/16 on the old set. This
inflates the apparent severity of the gap in F008 and partly explains why
Qwen3-4B measured only 25.0%-43.75% with thinking disabled, which was always
implausibly low for that model.

**F008's conclusion survives**: the arm is still underpowered at n=16
(+/-24 points), and that is a property of the sample size, independent of
whether the references are right. But its *observed accuracies* are now
superseded and cannot be used as a baseline for calibrating floors.

## What caught it

Sourcing the larger quality set for Tier 2 and cross-checking the new problems
against the existing pinned ones. Nothing in any run would have revealed it:
wrong ground truth produces a plausible low accuracy, not an error.

## Near-miss worth recording

The first cross-check keyed on a 90-character prompt prefix and reported three
answers as wrong that are in fact fine, because GSM8K contains near-duplicate
problems sharing long prefixes. That false result was briefly believed. It is
the same failure mode as F011 (AG_RS): a loose match producing a confident wrong
answer. Matching was redone on full normalised text against the complete split.

## Fix

The gate's GSM8K half is replaced with GSM8K test indices 0-15 verbatim,
verified against the published references, and the file records its source and
a version number. Prompt-set identity now participates in the staleness check
(see below), so results scored against the old set are re-run rather than
silently reused.

## How to reproduce

```bash
python -m specfp8.workloads.fetch_gsm8k --n 256
```
prints whether the gate head matches the published dataset.
