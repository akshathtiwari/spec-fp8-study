---
id: F037
title: The power-throttling hypothesis is refuted by direct measurement; both arms are power-capped 84-95% of the time and the FASTER arm is capped more, at a lower clock, drawing more power — the slow arm holds a higher boost clock while producing fewer tokens, which points at stalls rather than throttling
kind: result
status: established
confidence: high
date: 2026-10-04
evidence:
  runs: [2026-10-04T20-04Z_boot_variance, 2026-10-04T19-27Z_boot_variance]
  analysis: [specfp8/env.py]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

F035 proposed that the L4's CUDA-graph deficit is power throttling: a 72 W
part at 2040 MHz against a 150 W A10 at 1695 MHz, both at 22.5 GiB. It also
said the records could not test it, because `env` stores configured limits
rather than achieved values.

`GpuSampler` now samples achieved SM clock, power draw and the driver's own
`clocks_throttle_reasons.*` across each measurement window. One L4 session,
vLLM 0.29.0, 2 s interval:

| arm | n | tok/s | CV | **power-capped** | SM clock | power draw |
|---|---|---|---|---|---|---|
| default | 12 | **56.6** | 10.38% | **84%** of samples | 1991 MHz | 67.2 W |
| eager | 11 | **78.1** | 6.25% | **95%** of samples | **1645 MHz** | **69.8 W** |

The deficit is present and in the expected direction: ratio 0.725, consistent
with the 0.487 and 0.768 measured previously.

**The hypothesis is refuted, and the data runs opposite to it in three ways at
once.** The faster arm is power-capped *more* (95% against 84%), draws *more*
average power (69.8 W against 67.2 W), and runs at a *lower* SM clock
(1645 MHz against 1991 MHz). Thermal throttling is at **0.0%** in every sample
of both arms.

Power capping is not a differentiator here. It is a **constant**: this
configuration keeps a 72 W L4 at its cap for the overwhelming majority of
every run, in both arms. A quantity that is near-saturated in both conditions
cannot explain a 1.4x gap between them.

## What the data points at instead

The informative row is not the throttle fraction. It is this:

> The slow arm holds a **higher** boost clock (1991 MHz) while producing
> **fewer** tokens per second (56.6 against 78.1).

A card that is clocking up and delivering less work is not thermally or
power constrained in the way that matters. It is idle more often, or stalled,
or paying overhead that does not appear as compute. High boost clock with low
output is the signature of **gaps in the work**, not of a ceiling on it: the
governor sees short bursts separated by idle and boosts, where a dense steady
stream settles at a lower sustained clock and gets more done.

That is a hypothesis and this finding does not claim it. It is, however, a
better fit than the one it replaces, and unlike that one it is cheap to test:
`nvidia-smi` exposes `utilization.gpu`, which was not in the sampled field
set. If the default arm shows markedly lower GPU utilisation at a higher
clock, the stall reading is supported directly.

## What this does NOT establish

- **A mechanism.** Three candidates are now eliminated by evidence rather
  than left unexamined: graph mode and backend selection (F032), and power
  throttling here. None of that identifies what *is* happening.
- **That the L4 is not power-constrained at all.** It plainly is, 84–95% of
  the time in both arms. The claim is only that this does not distinguish
  the arms, and therefore cannot explain the difference between them.
- **That the clock difference causes anything.** Clock and throughput move in
  opposite directions here, which rules the simple reading out and does not
  establish the complicated one.
- **Anything about the A10.** The power comparison across cards that motivated
  F035 is untested: no A10 session has been run with sampling enabled. The
  cross-card reading in F035 is weakened by this result, not settled by it.
- **That sampling is free.** `GpuSampler` polls a subprocess every 2 s during
  the measurement. The interval and sample count are recorded so the cost is
  auditable, but it was not measured against an unsampled control.

## Why it matters that this was wrong

F035's capacity reading was retracted when both cards turned out to be
22.5 GiB — a field sitting unread in every record. Its replacement, power
throttling, survived four days and one instrument before dying too. Both were
plausible, both were stated as hypotheses rather than findings, and both were
wrong.

The thing that worked was not the reasoning. It was adding the instrument that
could falsify it and then running it against the claim. That is the fourth
time in this study that a confidently-held reading has been killed by making
the measurement rather than by thinking harder about the data already held.

The practice, which is now explicit three findings in a row:

> **When a hypothesis is proposed, the next action is the instrument that
> could refute it — not a better argument for it.**

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine bootvar --repeats 12 \
    --sweep-file sweeps/boot_stability.yaml
```

```bash
python -c "
import json, statistics as st, collections
recs=[json.loads(l) for l in open('results/probes.jsonl') if l.strip()]
g=collections.defaultdict(list)
for r in recs:
    d=r.get('gpu_during') or {}
    if not d.get('gpu_samples'): continue
    if (r.get('load') or {}).get('concurrency')!=1: continue
    g[r.get('arm')].append(d)
for arm,ds in sorted(g.items()):
    cap=[d['frac_power_capped'] for d in ds if d.get('frac_power_capped') is not None]
    clk=[d['sm_clock_mean_mhz'] for d in ds if d.get('sm_clock_mean_mhz') is not None]
    print(arm, 'capped', round(st.mean(cap)*100,1), '%  clk', round(st.mean(clk)))"
```
