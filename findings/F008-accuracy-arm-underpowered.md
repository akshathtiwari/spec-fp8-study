---
id: F008
title: The task-accuracy arm is statistically underpowered at n=16
kind: gap
status: established
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T04-32Z_h1]
  analysis: [analysis/out/tables/compatibility_matrix.md]
  code_sha: b294050
---

## Claim

Test 3's GSM8K accuracy check uses 16 prompts, giving a 95% confidence interval
of roughly **+/-24 percentage points** at p=0.40. It can detect a collapse to
near zero and nothing finer. No quality claim about FP8 is currently supportable.

## Evidence

Observed accuracies on Qwen3-4B, thinking disabled, spanning 25.0%–43.75% —
which is 4/16 versus 7/16, comfortably inside a single interval.

Power calculation (normal approximation, alpha=0.05, power=0.80):

| n | 95% CI half-width at p=0.40 |
|---|---|
| 16 | +/-24.0 pts |
| 100 | +/-9.6 pts |
| 250 | +/-6.1 pts |
| 400 | +/-4.8 pts |

To *detect* a 10-point drop: **356 per arm** treating arms as independent, or
**234 pairs** by McNemar at a 20% disagreement rate.

## Reasoning

The study already runs an identical pinned prompt set across every cell, which
is the paired design, so McNemar is the correct and more efficient test. The
architecture is right and merely undersized.

A second consequence: the inherited floors (0.70 correct / 0.50 degraded,
calibrated for a 4B target) marked every healthy Qwen3-4B server "broken",
because those floors were never validated against a measured baseline.

## What this does NOT establish

Whether FP8 actually damages quality. That is unmeasured, not measured-and-null.

## Remedy (Tier 2)

Expand GSM8K to ~250 pinned problems; separate the cheap per-cell gate from an
expensive quality measurement that can run at higher concurrency since task
accuracy, unlike exact match, does not need concurrency-1 determinism; set
floors from the measured baseline; report McNemar.
