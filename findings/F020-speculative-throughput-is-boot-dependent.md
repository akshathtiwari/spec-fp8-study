---
id: F020
title: Speculative throughput varies 2.16x across boots while acceptance does not
kind: result
status: established
confidence: high
date: 2026-09-22
evidence:
  runs: [2026-09-22T09-07Z_boot_stability, 2026-09-22T04-14Z_perf]
  analysis: [analysis/out/tables/throughput.md]
  code_sha: 1461e3a
supersedes: []
---

## Claim

On vLLM 0.29.0 / L4 (SM89), token throughput for a **speculative**
configuration varies by **2.16x** across boots of an identical configuration
with identical prompts, while the non-speculative control varies by **1.02x**
and acceptance (tau) varies by **1.08x**. The speculative path produces the
same accepted tokens per step every time and converts them to wall-clock
throughput at rates differing by more than a factor of two.

## Evidence

Six boots of each cell, `bf16` weights, `auto` KV, FLASHINFER, gsm8k, c=1,
n=16, seed 1234 — so prompts are identical throughout. Boots span three
containers, including three consecutive boots inside one container.

| | mean tok/s | s.d. | range | ratio |
|---|---|---|---|---|
| none (control) | 26.40 | 0.6% | 26.3 – 26.7 | **1.02x** |
| dflash | 46.94 | 29.3% | 28.4 – 61.2 | **2.16x** |

tau across the same six dflash boots: mean 4.183, range 4.091 – 4.411
(**1.08x**).

Per-request data confirms it is decode speed, not output length: for the same
cell, mean end-to-end latency was 3,505 ms in a fast boot against
7,316–7,923 ms in slow boots, with mean output lengths of 242 vs 250–265
tokens.

## Reasoning

The control is what makes this diagnostic. A 0.6% spread on the
non-speculative cell — across the same containers, the same GPU allocations
and the same prompts — excludes hardware, thermal state, container position
and prompt sampling as explanations. Whatever varies is specific to the
speculative path.

tau being stable while throughput is not also separates the two candidate
mechanisms: the drafter is accepting the same number of tokens per step, so
the variance is in the cost of producing them, not in how many are produced.

The engine warns at every speculative boot:

```
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
attention backend FlashInfer
```

and falls back to PIECEWISE capture. A CUDA-graph capture that differs between
boots is a plausible cause of a 2x decode-speed difference, but this finding
demonstrates the **effect**, not the cause. The cause is untested.

## What this invalidates

**The speculation x precision interaction reported from the Phase 2 grid is
withdrawn.** Speedups of 1.16x (bf16 weights) versus 2.71x (fp8 weights) were
each measured from a **single boot** per configuration, and a 2.16x boot-level
swing swamps that difference entirely.

More broadly, every DFlash throughput figure in `throughput.md` carries this
error term. Those cells have three repeats, but all three come from one boot,
so the +/-2-8% intervals shown measure prompt variance *within* a boot and are
silent about the dominant source of error.

## What this does NOT affect

- **F004 (KV capacity)** — capacity is read from the engine's startup log and
  reproduces exactly for a given compile-cache state.
- **F006 (tau invariance)** — tau reproduces to 1.08x across boots here and to
  <=0.002 within a boot.
- **F016 (accuracy)** — accuracy is stable across all six boots (93.8%).
- **F017 / F019 (compatibility, backend selection)** — unrelated to timing.

The non-speculative half of the Phase 2 grid is sound at 0.6% reproducibility.

## What this does NOT establish

- **A cause.** This finding measures the spread and shows acceptance is not
  responsible. It does not identify what varies. F021 answers this.
- **That the spread is continuous.** "Varies 2.16x" describes the range, not
  the distribution. Six boots cannot distinguish a wide unimodal spread from
  two tight modes, and reading it as continuous variance is what made the
  wrong hypothesis below look reasonable. It is bimodal (F021).
- **That 2.16x is the maximum.** It is the range observed in six boots of one
  configuration on one GPU. The eager-vs-default comparison in F021 later saw
  1.64x between just two boots at c=16, so six boots was not near saturation.
- **Anything about other engines, GPUs, or backends.** vLLM 0.29.0, L4/SM89,
  FLASHINFER, dflash only.

## Consequence for measurement design

Speculative throughput needs **boot-level repeats**, not prompt-level. Three
boots x three prompt draws per cell multiplies the dominant cost — boots — by
three: roughly 9 GPU-hours for the full grid.

Before paying that it is worth testing whether the variance is *controllable*
(for example by pinning `enforce_eager`, or by using a backend whose
speculative path does not fall back). A configuration that measures stably is
worth more than an average over an unstable one, and if the variance can be
removed the grid becomes cheaper rather than more expensive.

## Independent value

"Speculative decoding throughput on vLLM 0.29 / SM89 varies 2x across boots
while acceptance does not" is a reportable result on its own, and a warning to
anyone benchmarking speculative decoding from single runs — which, from the
prior art in requirements section 9, is the norm.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine sweep --sweep boot_stability --repeats 3
```

## Follow-up

The sharpest confirmation arrived later: cell `5824814b68901229` was
measured at **30.60 tok/s** in this grid and **68.33 tok/s** in perf_v2,
same cell_id, same argv, same arm, with tau moving 0.68% — below its own
noise floor. 2.23x on a configuration that is identical by construction.
See F021.


F021 resolves the mechanism: the variance is **bimodal, not continuous**, and
the mode is selected by the CUDA-graph path. `--enforce-eager` removes it —
boot spread falls from 1.64x to 1.04-1.07x while throughput rises 1.37-2.38x.
The hypothesis offered below (that graph *capture* varies per boot) is wrong;
each arm is individually stable, so whatever picks the mode is fixed at boot
and then holds.

The headline claim of this finding stands unchanged, and is what made F021
findable: acceptance is invariant to something that moves throughput 2x.
