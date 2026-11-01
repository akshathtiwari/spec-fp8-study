---
id: F040
title: Three further vLLM 0.29.0 sessions all reached parity, so the engine-release reading of the throughput split is withdrawn (3 deficit of 6 on 0.29.0 against 0 of 3 on later releases, Fisher p=0.238); what survives is a binary session-scoped regime whose ratio and dispersion partitions coincide exactly (p=0.012) and which separates all 116 individual boots without overlap
kind: result
status: established
confidence: high
date: 2026-10-31
evidence:
  runs: [2026-10-31T09-11Z_boot_variance, 2026-10-31T10-37Z_boot_variance,
         2026-10-31T07-30Z_boot_variance, 2026-10-31T06-59Z_boot_variance,
         2026-10-04T20-04Z_boot_variance, 2026-09-29T06-29Z_boot_variance,
         2026-09-23T19-28Z_boot_variance]
  analysis: [analysis/session_regimes.py, analysis/out/tables/session_regimes.md]
  code_sha: pending
supersedes: [F034, F038]
superseded_by: []
---

> **Revised by F042 (2026-11-01).** Every measurement below stands. Its
> interpretation does not. All three withdrawals in this finding -- the engine
> release (F034), L4 specificity (F035) and draft-model localisation (F018) --
> compared sessions on NVIDIA driver 580.95.05 against sessions on 610.57.04
> without knowing it. Within one driver, each original reading holds (3/3
> against 0/2, 0/1 and 0/2). And "session-scoped" is probably the wrong
> description: every deficit session shares one configuration (L4, 580.95.05,
> 0.29.0, DFlash), and every one-factor neighbour is at parity. This finding
> corrected an over-confident claim with an over-confident withdrawal.

## Claim

Two more vLLM 0.29.0 sessions were run on 2026-10-31 after F038, bringing
0.29.0 to six paired sessions. Both reached parity. The full table:

| engine | session | default | CV | eager | CV | ratio | regime |
|---|---|---|---|---|---|---|---|
| 0.29.0 | 09-23 | 39.2 | **13.92%** | 80.5 | 1.44% | **0.487** | deficit |
| 0.29.0 | 09-29 a | 61.2 | **3.73%** | 79.7 | 1.69% | **0.768** | deficit |
| 0.29.0 | 10-04 | 56.6 | **10.38%** | 78.1 | 6.25% | **0.724** | deficit |
| 0.29.0 | 10-31 a | 81.2 | 1.53% | 80.9 | 1.24% | 1.003 | parity |
| 0.29.0 | 10-31 b | 82.0 | 1.00% | 81.1 | 1.14% | 1.011 | parity |
| 0.29.0 | 10-31 c | 81.2 | 1.95% | 80.0 | 1.02% | 1.015 | parity |
| 0.30.0 | 09-29 b | 82.8 | 0.99% | 81.6 | 1.28% | 1.014 | parity |
| 0.30.0 | 09-29 c | 78.2 | 1.61% | 77.0 | 2.79% | 1.017 | parity |
| 0.32.0 | 10-31 d | 81.7 | 0.86% | 78.6 | 3.12% | 1.039 | parity |

Nine sessions, nine distinct physical cards, all reporting the same 22.5 GiB,
2040 MHz ceiling and 72 W limit.

### 1. The engine-release reading is withdrawn

F034 claimed the version difference "replicates as a ratio" on the strength
of two 0.29.0 sessions at 0.487/0.768 against two 0.30.0 sessions at
1.014/1.017. **Three further 0.29.0 sessions reached parity.** The tally is
3 deficit of 6 on 0.29.0 against 0 of 3 on later releases: one-sided Fisher
exact **p = 0.238**. At this sample size the releases are indistinguishable.

F038 had already weakened this to p=0.11 on four 0.29.0 sessions and
predicted that a further parity session would take it to roughly p=0.26. Two
arrived; the figure is 0.238. **The prediction was recorded before the runs,
which is the only reason this reads as a test rather than a rationalisation.**

### 2. What survives is stronger than the version claim ever was

- **Ratio** partitions into 0.487--0.768 and 1.003--1.039, a gap of **0.235**,
  wider than the spread inside either group.
- **Graph-arm dispersion** partitions into 3.73--10.38% and 0.86--1.95%,
  also disjoint.
- **The two partitions are identical.** Under random assignment the
  probability that the three highest-dispersion sessions of nine are exactly
  the three lowest-ratio ones is 1/84 = **0.012**.

So the deficit and the dispersion are not two findings. They are one
session-scoped phenomenon: a session is in one regime or the other, and in
the deficit regime it is both slow and unstable.

### 3. The separation holds boot by boot, not just session by session

Pooling every fixed-prompt default-arm boot at concurrency 1 gives **n=116**:
44 deficit boots spanning **29.5--65.6** tok/s, 72 parity boots spanning
**75.5--83.5**. **The groups do not overlap.** The gap is **9.9** tok/s,
against a largest within-group gap of **5.7**. Every individual boot measured
in a deficit session is slower than every individual boot measured in a
parity session.

This is the quantity a reader is exposed to. Running this configuration once
returns a draw from one of two separated populations, and nothing in the
configuration, the engine version or the recorded card identity says which.

### 4. Power throttling is refuted a second time, across regimes

F037 refuted it across the two arms of one deficit session. Instrumenting the
parity sessions gives the stronger comparison --- same arm, same
configuration, different regime:

| regime | session | tok/s | SM clock | capped | power | util |
|---|---|---|---|---|---|---|
| deficit | 10-04 | 56.6 | **1991 MHz** | 0.84 | 67.2 W | not sampled |
| deficit† | 10-04 b | 62.4 | **1985 MHz** | 0.91 | 68.7 W | not sampled |
| parity | 10-31 a | 81.2 | 1700 MHz | 0.96 | 70.1 W | 91.3% |
| parity | 10-31 b | 82.0 | 1808 MHz | 0.96 | 69.9 W | 91.4% |
| parity | 10-31 c | 81.2 | 1777 MHz | 0.95 | 70.0 W | 90.5% |
| parity | 10-31 d | 81.7 | 1714 MHz | 0.96 | 70.1 W | 90.5% |

†10-04 b ran no eager arm, so it has no ratio; its level and clock place it in
the deficit regime unambiguously. Thermal throttling is 0.00 everywhere.

**Deficit graph arms hold a clock 177--291 MHz higher than any parity graph
arm while producing 23--31% fewer tokens, drawing less power, and being
power-capped less often.** A card that boosts higher, draws less and delivers
less work is not against a ceiling. It is idle for part of the time.

### 5. Acceptance is quiet in both regimes

Nine sessions, three releases: mean tau spans 4.0702--4.1594 (a 2.2% range)
and no session disperses by more than **0.88%**. The contrast with throughput
is itself regime-dependent, which sharpens rather than weakens it: in deficit
sessions throughput dispersion exceeds tau dispersion by 12--22x; in parity
sessions both are below 2% and the factor falls to 1.6--3.8.

## Why this matters to the paper

The paper's second confound stops being "throughput is unreproducible and so
is its dispersion on 0.29.0" and becomes "throughput falls into one of two
session-scoped regimes, and which one is not predicted by anything we
record". That is a narrower claim about causes and a **much** stronger claim
about structure: a clean bimodal split over 116 boots with no overlap,
against a version comparison that no longer reaches significance.

It also costs the paper its tidiest story. "A regression, fixed in the next
release" is what a reader wants; it is not what the data says.

## What this does NOT establish

- **A mechanism.** Still unidentified. Graph mode and backend selection are
  ruled out by the engine log (F032), power throttling by direct sampling
  (F037 and §4 above). The stall reading remains untested.
- **That the engine release is irrelevant.** p=0.238 is a failure to
  establish an association, not evidence of none. 3 of 6 against 0 of 3 is
  the direction the version claim predicted; there is simply not enough of it.
- **That the release is separable from the calendar.** All three deficit
  sessions ran between 23 September and 4 October; four of six parity
  sessions, including all three 0.29.0 ones, ran on 31 October. Engine
  release and measurement date are entangled in this design. "0.29.0 was
  never the cause" and "whatever caused the deficit stopped occurring on this
  provider's fleet" both fit. Breaking it needs 0.29.0 and a current release
  interleaved within one day.
- **That the regime is a property of a card.** Nine sessions, nine distinct
  UUIDs, no card measured twice. The regime cannot be attributed to a
  specific card, and cannot be excluded as a property of card populations or
  of hosts.
- **That the A10 is immune** (retracting F035's reading). 0.959 is simply a
  parity value, slightly *below* the L4's own parity band of 1.003--1.039.
  Under the observed deficit frequency of 3 in 9, one session lands in parity
  by chance with probability 0.67. One A10 session is close to no evidence.
- **That the deficit regime is L4-specific.** Only that every deficit session
  observed was on an L4, having given other cards one session.
- **That the regime is binary rather than finely graded.** Nine sessions
  cannot exclude a third regime, or a continuum sampled sparsely. What is
  established is that these nine fall into two separated groups.
- **That n-gram speculation is exempt** (weakening F018). Both n-gram
  sessions sat in the parity regime. At the observed deficit frequency, two
  sessions both land in parity with probability (2/3)² = 0.44 even if n-gram
  is affected exactly as dflash is.

## Provenance

```
engine_version : honoured on every record (F031 guard)
gpu_uuid       : 9 distinct, one per session
telemetry      : clocks/power since 2026-10-04; utilization.gpu since 2026-10-31
```

The utilisation field was added on the last day of the quota window and no
deficit session has occurred since, so the one measurement that would test
the stall reading is the one we do not have. It is also the cheapest
experiment left in the study: one sweep, minutes of L4 time, gated only on
the regime appearing again.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine bootvar --repeats 12 \
    --sweep-file sweeps/boot_stability.yaml
```

```bash
python analysis/session_regimes.py
```
