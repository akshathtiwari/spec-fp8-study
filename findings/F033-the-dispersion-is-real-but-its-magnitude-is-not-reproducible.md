---
id: F033
title: Boot dispersion on vLLM 0.29.0 measured twice gives 13.92% and 3.73%, and mean throughput 39.2 and 61.2 tok/s, against an eager control that reads 80.5 and 79.7; the dispersion is real, its magnitude is not reproducible, and the paper reports one measurement of it as if it were the quantity
kind: result
status: established
confidence: high
date: 2026-09-29
evidence:
  runs: [2026-09-29T06-29Z_boot_variance, 2026-09-23T19-28Z_boot_variance,
         2026-09-29T04-48Z_boot_variance]
  analysis: [analysis/boot_statistics.py]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

The same experiment -- 12 boots of an identical DFlash configuration, vLLM
0.29.0, L4, bf16/auto, FLASHINFER, concurrency 1, one fixed prompt set, all
within a single container -- was run twice, six days apart:

| engine | arm | session | n | mean tok/s | CV |
|---|---|---|---|---|---|
| 0.29.0 | default | 2026-09-23 | 12 | 39.2 | **13.92%** |
| 0.29.0 | default | 2026-09-29 | 12 | 61.2 | **3.73%** |
| 0.29.0 | eager | 2026-09-23 | 8 | 80.5 | 1.44% |
| 0.29.0 | eager | 2026-09-29 | 12 | 79.7 | 1.69% |
| 0.30.0 | default | 2026-09-29 | 12 | 82.8 | 0.99% |
| 0.30.0 | eager | 2026-09-29 | 12 | 81.6 | 1.28% |

Two measurements of the same quantity, on the same engine version, differ by a
factor of **3.7 in dispersion** and **1.6 in level**.

**The eager arm is the control, and it barely moves.** 80.5, 79.7, 81.6 across
three sessions, two engine versions, and at least two distinct physical cards,
with CV between 1.28% and 1.69%. Whatever differs between sessions, it is not
something that moves the eager execution path.

Normalising the default arm by the eager arm measured in the same session
removes card and session level effects:

| session | default / eager |
|---|---|
| 0.29.0, 2026-09-23 | **0.49** |
| 0.29.0, 2026-09-29 | **0.77** |
| 0.30.0, 2026-09-29 | **1.01** |

On 0.29.0 the CUDA-graph path loses between 23% and 51% of the eager path's
throughput, and *how much it loses is not reproducible*. On 0.30.0 it reaches
parity.

So the paper's \S4 headline, CV 13.92%, is one session's measurement of a
quantity that a second measurement puts at 3.73%. The effect is real in both.
The number is not a property of the configuration.

## Two corrections to the analysis that produced this

Both were made while investigating, and both are recorded because each was
confidently asserted before being checked.

**First: this is not a session-pooling artifact.** An intermediate reading
held that the 13.92% pooled boots from three container sessions and therefore
mixed host variation into boot variation. A one-way decomposition refutes
that. All 12 boots behind the published figure come from a single session,
`2026-09-23T19-28Z`:

```
variance from BETWEEN sessions   0.0%
variance from WITHIN  sessions 100.0%
```

The three probe_runs visible in `results/probes.jsonl` span all mechanisms and
arms; the series behind the headline is one container. The published number is
genuine within-session boot dispersion, as \S4 says it is.

**Second: the 0.30.0 comparison was reported as host-confounded and is less
so than stated.** F032 noted that the 0.30.0 run is one session while the
baseline was believed to be three. Since the baseline is also one session,
both are single-session measurements; they ran on different physical cards,
which the eager normalisation above largely absorbs.

## Why it happened

The study measured dispersion once per arm and treated the result as the
dispersion. That is the same error the paper documents in others: reporting a
quantity from a single draw. It is sharper here because the quantity *is* a
dispersion, so the omission is not a missing error bar on a mean but a missing
error bar on an error bar.

Nothing prompted a second measurement, because the first produced a clean,
publishable number with n=12 and a plausible mechanism story. A result that
looks finished does not invite replication, which is F028's observation
arriving in the headline rather than in a checker.

## What this does NOT establish

- **That either measurement is wrong.** Both are 12 fixed-prompt boots in a
  single container, both reproduce from stored records, and both show the
  default arm dispersing more than the eager arm measured beside it.
- **What differs between the two sessions.** Different physical card, six days
  apart, and the second ran newer harness code. GPU identity was not recorded
  on 2026-09-23, so the cards cannot even be compared. This is precisely the
  gap `gpu_uuid` was added to close, and it was added one day too late to
  close it retrospectively.
- **That 0.30.0 fixed anything.** F032 rules out the two checkable mechanisms
  -- the CUDA-graph fallback still fires on 0.30.0, and the backend split is
  unchanged. A single 0.30.0 session showing 0.99% is one draw of a quantity
  this finding shows is not reproducible from one draw. **0.30.0 has not been
  measured twice.**
- **That dispersion decreases monotonically with version.** Three points,
  one per session, of a quantity whose session-to-session spread is the thing
  being reported.

## The new data silently moved three published figures

Adding the two new sessions to `results/probes.jsonl` changed derived numbers
that the paper quotes, with nothing failing until `check_paper.py` ran:

| figure | published | after the new runs | cause |
|---|---|---|---|
| DFlash default CV | 13.92% | 23.85% (n=24) | two 0.29.0 sessions pooled |
| acceptance CV | 0.62% (n=12) | 0.54% (n=24) | control session pooled into `_probe_taus` |
| pooled-with-prompts CV | 29.92% (n=23) | 26.35% (n=35) | new records entered the contrast set |

The first is the most instructive. Pooling two sessions whose CVs are 13.92%
and 3.73% yields **23.85%, higher than either**, because their means differ
(39.2 and 61.2) and between-session spread enters the pooled variance. A
pooled dispersion is not an average of dispersions.

An earlier commit added `engine_ref` and `gpu` filters to `boots()` precisely
to stop this, and it was not enough: it guarded one function while
`_probe_taus()` and the pooled-contrast section next to it were left open, so
two of the three leaks were in code written to prevent the leak. All three
series are now pinned to the published measurements via `has_uuid`, and the
per-session tables report everything else.

This is why the paper's numbers are regenerated by a script that the build
refuses to pass rather than transcribed. Three published figures moved, none
of them announced it, and the only reason it was caught is that a checker
recomputes every number in the paper from the records on every build.

## Consequence

\S4 must report both measurements rather than one, and must say that the
magnitude did not reproduce. The defensible claims are:

1. The default CUDA-graph path disperses more than the eager path measured
   beside it, in every session tested (13.92 vs 1.44; 3.73 vs 1.69).
2. It is also *slower* than the eager path on 0.29.0, by 23% to 51%.
3. Both the dispersion and the deficit vary between sessions by more than a
   factor of three, so neither should be quoted as a single number.
4. The eager arm is stable across sessions, versions and cards, which is what
   makes it usable as a control.

The practice this forces:

> **A dispersion measured once is a draw, not a property.** If the headline
> number is a variance, it needs replication for the same reason a mean does,
> and this study did not do that for four revisions.

By the paper's own standard the honest headline is not "throughput disperses
by 13.92% across boots" but "throughput disperses across boots by an amount
that itself varies between sessions, from 3.73% to 13.92%, while the eager
control stays near 1.5%."

**0.30.0 now needs a second session** before F032's 0.99% can be quoted the
way 13.92% was quoted. Running it once and believing it is the error this
finding is about.

## How to reproduce

```bash
python - <<'EOF'
import json, statistics as st, collections
recs=[json.loads(l) for l in open('results/probes.jsonl') if l.strip()]
rows=collections.defaultdict(list)
for r in recs:
    c=r.get('config') or {}
    if r.get('phase')!='boot_variance' or c.get('mechanism')!='dflash': continue
    if (r.get('load') or {}).get('concurrency')!=1: continue
    v=(r.get('throughput') or {}).get('output_tokens_per_s')
    if v: rows[(c.get('engine_ref'), r.get('arm'), r.get('probe_run'))].append(v)
for k,v in sorted(rows.items()):
    if len(v)>1:
        print(k, len(v), round(st.mean(v),1), round(st.stdev(v)/st.mean(v)*100,2))
EOF
```
