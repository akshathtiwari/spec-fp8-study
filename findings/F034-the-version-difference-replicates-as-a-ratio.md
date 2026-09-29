---
id: F034
title: Measured twice on different cards, vLLM 0.30.0 puts the CUDA-graph path at parity with eager (ratio 1.014 and 1.017) where 0.29.0 puts it at 0.487 and 0.768; the eager-normalised ratio replicates where neither the raw throughput nor the raw dispersion does
kind: result
status: established
confidence: high
date: 2026-09-30
evidence:
  runs: [2026-09-29T17-16Z_boot_variance, 2026-09-29T04-48Z_boot_variance,
         2026-09-29T06-29Z_boot_variance, 2026-09-23T19-28Z_boot_variance]
  analysis: [analysis/boot_statistics.py]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

F032 reported that the boot dispersion present on vLLM 0.29.0 was absent on
0.30.0, and F033 immediately disqualified that from being quoted: it was one
session, and F033 had just shown that a single-session dispersion is a draw
rather than a property. A second 0.30.0 session, on a different physical card,
now exists.

| engine | arm | session | card | n | mean | CV |
|---|---|---|---|---|---|---|
| 0.29.0 | default | 09-23 | not recorded | 12 | 39.2 | **13.92%** |
| 0.29.0 | default | 09-29 06-29 | `GPU-7ad44181` | 12 | 61.2 | **3.73%** |
| 0.30.0 | default | 09-29 04-48 | `GPU-e08becdf` | 12 | 82.8 | **0.99%** |
| 0.30.0 | default | 09-29 17-16 | `GPU-15116537` | 12 | 78.2 | **1.61%** |
| 0.29.0 | eager | 09-23 | not recorded | 8 | 80.5 | 1.44% |
| 0.29.0 | eager | 09-29 06-29 | `GPU-7ad44181` | 12 | 79.7 | 1.69% |
| 0.30.0 | eager | 09-29 04-48 | `GPU-e08becdf` | 12 | 81.6 | 1.28% |
| 0.30.0 | eager | 09-29 17-16 | `GPU-15116537` | 11 | 77.0 | 2.79% |

**0.30.0's low dispersion replicates; 0.29.0's high dispersion does not.**
0.99% and 1.61% against 13.92% and 3.73%. Three distinct physical cards are
involved and one session's card was never recorded.

**The robust statistic is the ratio, not either raw number.** Dividing each
default arm by the eager arm measured beside it in the same container removes
session and card level entirely:

| engine | session | default / eager |
|---|---|---|
| 0.29.0 | 09-23 | **0.487** |
| 0.29.0 | 09-29 | **0.768** |
| 0.30.0 | 09-29 04-48 | **1.014** |
| 0.30.0 | 09-29 17-16 | **1.017** |

The two 0.30.0 sessions agree to within **0.3%** on different cards. The two
0.29.0 sessions disagree by **58%**. So the defensible version claim is not
about dispersion at all: on 0.29.0 the CUDA-graph path costs between 23% and
51% of eager throughput and the cost is unpredictable; on 0.30.0 it is at
parity, reproducibly.

## The eager control did its job

The second 0.30.0 session ran 5.6% slower than the first on *both* arms:
default 82.8 to 78.2, eager 81.6 to 77.0. Because both moved together, the
eager arm identifies this as a property of the session or card rather than of
the execution path under study, and the ratio absorbs it (1.014 to 1.017).

Without a same-session control this would have looked like the 0.30.0 result
failing to replicate by 5.6%. With it, the level shift is attributable and the
quantity of interest is unchanged. This is the case for running a control arm
inside every session rather than comparing to a remembered number.

## Provenance

```
engine_version : {'0.30.0': 68}        all records in the new session
engine_check   : {'honoured': 68}      declared engine_ref matched the server
gpu_uuid       : GPU-15116537-1561-1668-4109-01b3be07c657  (1 distinct)
```

The F031 guard passed again, and the card differs from the first 0.30.0
session, which is what makes this a replication rather than a repeat.

## What this does NOT establish

- **A mechanism.** F032 rules out the two checkable candidates: the
  `FULL_AND_PIECEWISE` fallback still fires on 0.30.0 and capture still runs
  under `PIECEWISE`, and the FLASHINFER/FLASH\_ATTN backend split is
  unchanged. Why parity is reached is unknown.
- **That 0.29.0's dispersion is bimodal, or has any structure.** Two sessions
  give 13.92% and 3.73%. Two points do not describe a distribution, and F029
  retracted a structure claim made from five.
- **That the ratio is stable on 0.29.0 at n=2.** 0.487 and 0.768 differ by
  58%; more sessions could widen or narrow that. What is established is that
  it is *not* near 1.0 on 0.29.0 and *is* on 0.30.0.
- **Anything about other hardware, models, mechanisms or concurrencies.**
  Four sessions, one card model, one target, one drafter, concurrency 1.
- **That the second session is complete.** It hit the 90-minute function
  timeout after 12 default boots and 11 of 12 eager boots. The eager arm is
  therefore n=11. Records are written per measurement, so the truncation costs
  one boot rather than the run.

## Consequence

\S4 can now make a replicated claim, and should make it about the ratio:

> On vLLM 0.29.0 the default CUDA-graph path delivers 49% and 77% of the
> throughput of the same configuration under `--enforce-eager`, measured in
> two sessions. On 0.30.0 it delivers 101% and 102%. The deficit is real,
> its size is not reproducible, and it is gone in the following release.

And the general practice, which is the transferable part:

> **Put the control in the same container as the treatment.** Every number in
> this study that survived scrutiny is a within-session comparison; every
> number that had to be withdrawn or restated was a cross-session one. The
> eager arm is only useful because it was measured beside the default arm,
> not because it was measured.

## How to reproduce

```bash
SPECFP8_VLLM_VERSION=0.30.0 modal run cloud/modal_probe.py \
    --engine bootvar --repeats 12 \
    --sweep-file sweeps/boot_stability_v030.yaml
```

The per-session table in `analysis/out/tables/boot_statistics.md` carries
every session; the ratio is computed from the `default` and `enforce_eager`
means sharing a `probe_run`.
