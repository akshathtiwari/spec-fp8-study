---
id: F038
title: A fifth vLLM 0.29.0 session produced parity (ratio 1.003), so the deficit is intermittent rather than a property of the release; deficit and dispersion are perfectly correlated and the two regimes are disjoint, which makes this one session-scoped phenomenon rather than two boot-level ones, and weakens F034's version claim to p=0.11
kind: result
status: established
confidence: high
date: 2026-10-31
evidence:
  runs: [2026-10-31T07-30Z_boot_variance, 2026-10-31T06-59Z_boot_variance]
  analysis: [analysis/boot_statistics.py]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

F034 reported that the version difference "replicates as a ratio": 0.487 and
0.768 on vLLM 0.29.0 against 1.014 and 1.017 on 0.30.0. A fifth 0.29.0
session, run today with the identical sweep on an L4, gives **1.003**.

All L4 sessions to date, default arm divided by the `--enforce-eager` arm
measured in the same container:

| engine | session | ratio | default CV | regime |
|---|---|---|---|---|
| 0.29.0 | 09-23 | 0.487 | 13.92% | degraded |
| 0.29.0 | 09-29 | 0.768 | 3.73% | degraded |
| 0.29.0 | 10-04 | 0.724 | 10.38% | degraded |
| **0.29.0** | **10-31** | **1.003** | **1.53%** | **healthy** |
| 0.30.0 | 09-29 b | 1.014 | 0.99% | healthy |
| 0.30.0 | 09-29 c | 1.017 | 1.61% | healthy |
| 0.32.0 | 10-31 | 1.039 | 0.86% | healthy |

Three things follow, and they do not all point the same way.

**1. The deficit is intermittent on 0.29.0, not characteristic of it.** Three
of four L4 sessions show it; one does not. "vLLM 0.29.0 has a CUDA-graph
deficit" is the wrong shape of claim. The right one is that a deficit appears
in some sessions and not others.

**2. The deficit and the dispersion are the same phenomenon.** Every session
has both or neither, and the two regimes are **disjoint**: degraded sessions
span CV 3.73-13.92%, healthy ones 0.86-1.61%, with no overlap. The
probability that the three degraded sessions are exactly the three highest
CVs by chance is 1/C(7,3) = **p = 0.029**. This study has been reporting a
"deficit" and a "dispersion" as two findings. They are one.

**3. It is session-scoped, not boot-scoped.** Within a degraded session every
boot is slow and variable; within a healthy session every boot is fast and
stable. Whatever determines the regime is fixed once per container and
inherited by all twelve boots inside it. That is a materially different claim
from "throughput varies across boots", which is how \S4 has described it from
the first revision.

## What this does to F034

F034's version claim is weaker than F034 states. The contingency is 3
degraded of 4 on 0.29.0 against 0 of 3 on 0.30.0 and 0.32.0. A one-sided
Fisher exact test gives **p = 0.114**.

So: the two regimes are real (p = 0.029), and the proposition that the engine
release determines which one you get is **not established** (p = 0.114). With
seven sessions it cannot be. F034 should not have read 0.487/0.768 against
1.014/1.017 as a version effect when n was two and two.

The honest statement is that no degraded session has been observed on 0.30.0
or later, across three sessions, and that this is consistent with a fix and
also consistent with a roughly one-in-four event not yet having occurred.

## The stall hypothesis is still untested

Today's 0.29.0 session carried `utilization.gpu` for the first time,
specifically to test F037's surviving reading: that the slow arm holds a
higher clock while producing fewer tokens because it is stalled rather than
capped.

It cannot test that, because **no deficit appeared.** Utilisation was 91.3%
on the default arm and 91.6% on eager, with the ratio at 1.003 -- a healthy
session, which has no stall to observe. An instrument aimed at a phenomenon
that did not occur has measured nothing about it.

The test still requires a *degraded* session with utilisation sampling, and
degraded sessions are the ones we cannot summon on demand. Two further 0.29.0
sessions are queued for exactly this reason; at roughly three in four, one of
them is likely to degrade.

## vLLM 0.32.0

Measured today for version currency: ratio 1.039, default CV 0.86%,
utilisation 90.5% against eager's 89.7%. Healthy on every axis, and the
lowest dispersion recorded in the study.

One caveat that will not go away. vLLM 0.31.0 lists as a breaking change that
`--enforce-eager` now also disables JIT kernel warmup. The eager arm is our
control, so from 0.31.0 onward it is not the same intervention it was in
0.29.0 and 0.30.0, and the 1.039 is not strictly comparable to the earlier
ratios. Suggestively, eager's CV on 0.32.0 is 3.12% against 1.28% and 2.79%
on 0.30.0 -- the control arm itself got noisier, which is what one would
expect if it now does more at boot.

## What this does NOT establish

- **That 0.30.0 did not fix anything.** Zero degraded sessions in three is
  weak evidence for a fix, not evidence against one.
- **A mechanism.** Four candidates have now been eliminated or left untested:
  graph mode and backend selection (F032), power throttling (F037), and the
  stall reading (untested, this finding). The FlashInfer autotune caching
  added in 0.32.0 is a new candidate and is also untested.
- **That the regimes are truly bimodal.** Seven sessions, three in one group.
  F029 retracted a bimodality claim made from five boots, and the same
  caution applies here with more force: disjoint ranges at n=7 are suggestive
  of structure, not proof of it.
- **That the rate is 3 in 4.** Four 0.29.0 sessions give a point estimate of
  0.75 with a 95% interval running from roughly 0.19 to 0.99. The rate is
  almost unconstrained.

## Consequence

\S4 needs restating a third time, and the restatement is structural rather
than numerical:

> Sessions fall into two regimes. In a degraded session the CUDA-graph path
> runs at 49-77% of the eager path and disperses at 3.7-13.9% across boots;
> in a healthy session it is at parity and disperses below 1.6%. Every
> session is wholly one or the other. Three of four 0.29.0 sessions were
> degraded and none of three on 0.30.0 or later; with those numbers the
> engine release is a plausible but unestablished cause.

The practice this forces, which is the fourth variation on the same theme:

> **A phenomenon that appears in most trials is not a property of the
> configuration until you have run enough trials to see it fail to appear.**
> Four sessions told us the deficit was characteristic of 0.29.0. The fifth
> told us it is not.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine bootvar --repeats 12 \
    --sweep-file sweeps/boot_stability.yaml
```

```bash
python -c "
from math import comb
print('version claim p =', round(sum(comb(3,x)*comb(4,4-x)/comb(7,4) for x in (3,)),3))
print('ratio/CV association p =', round(1/comb(7,3),3))"
```
