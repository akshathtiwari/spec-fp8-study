---
id: F043
title: No engine log message separates the deficit configuration from its neighbours (0 of every distinct template across 140 graph-arm logs); three startup candidates are eliminated; and F042's n-gram neighbour was not one factor but three -- n-gram silently runs the V1 model runner with async scheduling disabled -- so it cannot localise the deficit to DFlash
kind: result
status: established
confidence: high
date: 2026-11-01
evidence:
  runs: [2026-09-23T19-28Z_boot_variance, 2026-09-29T06-29Z_boot_variance,
         2026-10-04T20-04Z_boot_variance, 2026-10-04T19-27Z_boot_variance,
         2026-09-29T04-48Z_boot_variance, 2026-09-29T17-16Z_boot_variance,
         2026-10-31T07-30Z_boot_variance, 2026-10-31T09-11Z_boot_variance,
         2026-10-31T10-37Z_boot_variance, 2026-11-01T12-30Z_boot_variance,
         2026-09-23T07-20Z_boot_variance, 2026-09-23T11-08Z_boot_variance,
         2026-09-30T03-16Z_boot_variance]
  logs: [results/logs/]
  analysis: [analysis/engine_logs.py, analysis/out/tables/engine_logs.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

Every engine log for every session in F042's configuration table had been
written to the Modal volume and never read. 315 of them were not in the repo.
They are now in `results/logs/`, and `analysis/engine_logs.py` reads them,
deriving each log's cell from `results/probes.jsonl` rather than from a list
typed by hand. Three results.

### 1. The deficit is not a logged event

Every distinct message template across 140 graph-arm logs, normalised for
numbers, timestamps and hashes, was compared between the deficit cell
(L4, 580.95.05, 0.29.0, DFlash) and its one-factor neighbours:

- present in at least 75% of deficit logs and at most 10% of every
  neighbour: **0**
- present in at most 10% of deficit logs and at least 75% of every
  neighbour: **0**

No capture fallback, recompilation, eager fallback or warning appears in the
deficit cell and not elsewhere. Every warning the deficit logs carry, every
parity neighbour on DFlash carries too, in every log. The engine runs the same
path, prints the same messages, and is slower. Whatever the deficit is, vLLM
does not know it is happening.

### 2. Three startup candidates are eliminated

| cell | first capture, s per graph | graph pool, GiB | JIT kernels during inference |
|---|---|---|---|
| deficit | 0.35--53.86 | 0.15 | 4 kinds in 44/44 logs |
| engine 0.30.0 | 0.33--53.02 | 0.37 | 4 kinds in 24/24 logs |
| driver 610.57.04 | 0.22--45.61 | 0.15 | 4 kinds in 48/48 logs |
| A10 | 0.21--26.96 | 0.15 | 4 kinds in 12/12 logs |

- **Triton JIT compilation during inference.** The deficit logs warn that
  `_prepare_dflash_inputs_kernel`, `_compute_local_logits_stats_kernel`,
  `_rejection_kernel` and `_resample_kernel` compile mid-inference. So do
  every DFlash parity neighbour's, in every log. A DFlash property, not a
  regime property.
- **Slow first graph capture.** Up to 53.86 s for one graph in the deficit
  cell -- and 53.02 s on 0.30.0, 45.61 s on 610.57.04, both at parity. It is
  the first boot in a container compiling before the torch.compile cache is
  warm, and it happens in both regimes.
- **Graph pool size.** 0.30.0 allocates 0.37 GiB against 0.15 GiB in every
  0.29.0 cell. A real difference between releases, and not the cause: the
  driver neighbour, at parity, uses the same 0.15 GiB as the deficit.

### 3. The n-gram neighbour was three factors, not one

| cell | model runner | async scheduling |
|---|---|---|
| deficit | V2 | on |
| engine 0.30.0 | V2 | on |
| driver 610.57.04 | V2 | on |
| A10 | V2 | on |
| **n-gram** | **V1** | **off** |

Every n-gram log carries two warnings no DFlash log does:

> Model Runner V2 does not yet support ngram/ngram_gpu speculative decoding;
> using the V1 model runner instead.

> Async scheduling not supported with ngram-based speculative decoding and
> will be disabled.

So switching DFlash to n-gram changed the speculation mechanism, the model
runner and async scheduling at once. F042 listed it as a one-factor neighbour
and read its parity as evidence the deficit needs DFlash. It is equally
evidence the deficit needs the V2 runner, or async scheduling, or any
combination. **The n-gram comparison cannot localise the deficit to DFlash.**

This is the study's first confound (F001) in a new place: change one setting
and the engine silently substitutes a component. There it was the attention
backend; here it is the model runner.

The 2x2 is unaffected. `none` runs V2 with async scheduling on, exactly as
DFlash does (24 of 24 logs), so a `none` arm is a clean speculation-only
control. The n-gram one never was.

## The hypothesis this produces

Not a finding; recorded so that the next experiment can be designed against
it.

**Async scheduling, under CUDA graphs, on driver 580.95.05.** Async
scheduling exists to keep the GPU busy by overlapping the CPU-side scheduler
with GPU execution. If that overlap breaks -- a synchronisation the driver
forces, a graph replay that blocks -- the GPU waits on the CPU between steps.
That is precisely the deficit's telemetry from F040: the slow arm holds a
*higher* SM clock, draws *less* power and is capped *less* often while
delivering fewer tokens. A card idle part of the time looks like that. And
the only configuration that escaped the deficit with the mechanism changed
also had async scheduling turned off.

It fits, it is specific, and it is not shown. Three things it would need:

- the deficit present with async scheduling on and absent with it off, on
  driver 580.95.05, everything else held fixed
- the V2 runner ruled in or out separately, since n-gram changed both
- GPU utilisation sampled inside a deficit session showing idle gaps, which
  F040's utilisation baseline (90.5--91.6% at parity) can be compared against

## What this changes about the intervention

The driver-swap experiment proposed in F042 gains an arm. On driver
580.95.05, run DFlash twice: once as is, once with async scheduling disabled.

| driver | async | prediction under this hypothesis |
|---|---|---|
| 580.95.05 | on | deficit |
| 580.95.05 | off | **parity** |
| 610.57.04 | on | parity |

If the second row comes back at parity, the deficit needs async scheduling,
and the mechanism question narrows from "something in a driver" to one
scheduler feature. If it comes back deficient, async scheduling is cleared
and the V2 runner is the next suspect.

## What this does NOT establish

- **That async scheduling causes the deficit.** It is the hypothesis the
  logs make most specific, not one they test. Nothing here varied async
  scheduling while holding the mechanism fixed.
- **That the logs are complete.** vLLM logs what its authors chose to log.
  "Not a logged event" means the engine reports nothing different, not that
  nothing different happens. A stall in the CUDA driver would be invisible
  here by construction.
- **That DFlash is irrelevant.** The n-gram contrast is uninformative about
  DFlash; it is not evidence against it. F042's DFlash row stands
  withdrawn, not reversed.
- **Anything about steady-state execution.** These are startup and message
  facts. The deficit lives in steady state, which these logs record only as
  periodic throughput lines.

## Provenance

```
logs added to results/logs/   : 315, from the Modal volume
logs already committed        : 27; 17 identical to the volume copy,
                                10 cell-id-named grid logs that later runs
                                overwrote on the volume -- left untouched,
                                since replacing committed evidence with a
                                newer file would change what cites it
scanned before committing     : 342 files, no token-, key- or email-shaped
                                strings
```

## How to reproduce

```bash
python analysis/engine_logs.py
```
