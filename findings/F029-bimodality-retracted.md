---
id: F029
title: RETRACTED — speculative throughput is not bimodal; it is one wide continuous distribution, and the two modes were a five-sample illusion
kind: retraction
status: retracted
confidence: high
date: 2026-09-24
evidence:
  analysis: [analysis/out/tables/figure1_data.md, analysis/boot_modes.py]
  code_sha: pending
supersedes: [F021]
superseded_by: []
---

## What was believed

F021 claimed that speculative throughput is **bimodal**: that boots land in
either a slow mode near 30 tok/s or a fast mode near 60–70, that the mode is
selected by the CUDA-graph path and fixed at boot, and that
`--enforce-eager` works by reliably selecting the fast mode. The paper's §4
was built on it. It was the study's headline.

## What is true

Pooling every valid DFlash default-arm boot at concurrency 1 across all
sessions, **n = 23**:

```
24.8 28.4 29.5 31.9 32.1 33.2 37.0 39.1 40.6 41.1 41.3 43.7
44.2 44.5 45.5 47.3 50.7 60.8 61.2 62.0 64.7 66.5 73.6
```

That is a **continuously filled distribution**, spread 2.96x. Fitting one
versus two Gaussian components:

| model | BIC |
|---|---|
| 1 component | 190.50 |
| 2 components | 192.10 |

$\Delta$BIC = **-1.60**, favouring a **single component**. The separation
test in `analysis/boot_modes.py` agrees: largest adjacent gap 0.53x the
within-group scatter, against a 3.0x threshold.

The twelve boots taken in one container on 2026-09-24 make it plainest —
29.48, 31.86, 32.06, 37.02, 39.06, 40.56, 41.06, 41.34, 43.71, 44.21, 44.51,
45.51 — a smooth ramp with no gap anywhere.

## Why it was believed

F021's founding evidence was five boots: **28.4, 33.2** against **50.7,
60.8, 61.2**. At n=5 a gap in the middle looks like separation. It is not;
it is what a wide distribution looks like when thinly sampled. The
intervening values, 37–50, simply had not been drawn yet. Every one of them
has now been observed.

This is **precisely the error F020 made and F021 claimed to correct**. F020
read a 2.16x range as continuous variance; F021 "corrected" it to bimodal;
the correction was wrong and the original reading was closer to right. The
claim inverted twice on samples too small to support either verdict.

## How it was caught

Not by inspection. `analysis/boot_modes.py` implements a separation
criterion for exactly this question, was written to test n-gram, and was
never pointed at the DFlash data — see F028. Pointed there, it failed
immediately. The instrument existed for a day before anyone aimed it at the
claim it was designed to check.

## What survives, and is stronger for being simpler

The retraction removes a mechanism, not the result:

- **Throughput of an identical speculative configuration varies 2.96x
  across 23 boots** while acceptance does not move. Well-powered, and the
  striking part was never the mode structure.
- **`--enforce-eager` collapses that variance to 1.04x** (n=8, 78.3–81.6)
  and lands **above every one of the 23 default boots**. In a single
  container on 2026-09-24: default 29.5–45.5 across 12 boots, eager
  78.3–81.6 across 8.
- **n-gram shows neither** (1.10x over 12 boots, F027).
- **tau is unmoved throughout**, which is what made the whole thing visible.

Dropping "bimodal" costs nothing a practitioner needed. "This configuration's
throughput is unreproducible to within a factor of three unless you disable
CUDA graphs" is the actionable claim, and it does not depend on the shape of
the distribution.

## What this does NOT establish

- **That the distribution is unimodal.** $\Delta$BIC of -1.60 is weak
  evidence for one component, not proof of it. The honest statement is that
  two components are **not supported at n=23**, which is different from
  showing there is one. Making the opposite mistake in the opposite
  direction is how this claim got here.
- **That the CUDA-graph attribution is wrong.** Eager removes the variance
  and raises throughput; pinning `cudagraph_mode: PIECEWISE` reproduces the
  low end. That evidence is untouched. What is withdrawn is the claim of a
  *discrete fallback state*.
- **That boots are independent draws.** Boots within one session span
  29.5–45.5 while the pooled range is 24.8–73.6, which hints at a
  session-level component. Not measured, and not claimed either way.

## Consequence

F021 is superseded. The paper's §4 is rewritten to report variance rather
than modes, its title and abstract lose the word, and Figure 1 now plots
every boot so the filling-in is visible rather than hidden behind five
points.

The study's own thesis applies to itself one more time: an effect was read
off a sample too small to support it, survived because nobody re-tested it,
and was caught by an instrument that had been sitting unused. That is the
fourth time in this project, and it is the most consequential.
