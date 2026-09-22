---
id: F021
title: Speculative throughput is bimodal; enforce_eager reliably selects the fast mode and removes the boot variance, but its throughput advantage depends entirely on which mode the comparison arm drew
kind: result
status: established
confidence: high
date: 2026-09-22
evidence:
  runs: [2026-09-22T12-1xZ_boot_variance_conc, 2026-09-22T10-0xZ_boot_variance,
         2026-09-22T09-07Z_boot_stability]
  analysis: []
  code_sha: 624ffdb
supersedes: []
---

## Claim

The 2.16x spread in speculative throughput (findings/F020) is **not
continuous variance**. It is **bimodal**: boots land in either a slow mode
near 30 tok/s or a fast mode near 58-61 tok/s. Disabling CUDA graphs with
`--enforce-eager` reliably selects the fast mode, making speculative decoding
**~1.9x faster** at identical acceptance.

## Evidence

Three arms, three boots each, one container, identical prompts, dflash /
bf16 / auto KV / FLASHINFER / gsm8k c=1 n=16:

| arm | mean tok/s | s.d. | min - max | ratio | tau |
|---|---|---|---|---|---|
| default | 30.57 | 4.3% | 29.07 - 31.50 | 1.08x | 4.09-4.12 |
| **enforce_eager** | **58.33** | 5.9% | 54.37 - 60.65 | 1.12x | 4.143 |
| piecewise (pinned) | 30.04 | 6.0% | 28.09 - 31.69 | 1.13x | 4.077 |

Re-reading the six boots behind F020 in this light, they cluster rather than
scatter: **28.4, 33.2** against **50.7, 60.8, 61.2**. Two groups, not a
spread.

## Reasoning

The hypothesis in F020 — that CUDA-graph *capture* varies per boot — is
**wrong**. Every arm here is individually stable (1.08-1.13x), including the
default arm that produced both modes across earlier runs. Whatever selects
the mode is fixed early and then holds.

What the test does establish is the direction: turning CUDA graphs **off**
nearly doubles speculative throughput. That is consistent with the warning
the engine emits at every speculative boot —

```
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
attention backend FlashInfer
```

— followed by a fallback that evidently costs more than it saves here.
Pinning `cudagraph_mode: PIECEWISE` explicitly reproduces the slow mode
exactly (30.04 vs 30.57), which indicates the fallback *is* the slow mode
rather than being a third state.

tau is unchanged across all arms (4.08-4.14), so this is purely the cost of
producing accepted tokens, not how many are accepted — the same separation
F020 established.

## The effect holds under load, and eager is the *stable* arm

The concurrency check (2 boots x 2 arms x c in {1, 16, 64}, one container,
identical prompts, same cell):

| conc | default | boot spread | eager | boot spread | eager/default |
|---|---|---|---|---|---|
| 1 | 31.1 | 1.02x | 74.2 | 1.18x | **2.38x** |
| 16 | 464.3 | **1.64x** | 776.7 | 1.07x | **1.67x** |
| 64 | 1203.8 | 1.18x | 1648.2 | 1.04x | **1.37x** |

The ratio never inverts. It *compresses* with load — 2.38x to 1.37x —
consistent with CUDA graphs recovering some of their value at larger batch
sizes, but they do not catch up within the range measured.

The second column matters more than the first. The **default** arm varies
**1.64x between two boots of an identical configuration** at c=16; the eager
arm varies 1.07x. So the bimodality of F020 is a property of the CUDA-graph
path, not of the GPU, the container, or the prompts. Turning graphs off does
not merely make speculative decoding faster — it makes it *measurable*.

tau is flat at 4.06-4.21 across all twelve measurements, so nothing here
touches acceptance.

## Independent corroboration from pre-hypothesis data

Re-reading the Phase 2 grid — collected on 2026-09-22 **before** the
bimodality hypothesis existed — turns up the same effect with a control
attached. Four configurations were each measured three times with the
repeats spread across two boots hours apart:

| cell | conc | spread across boots | gap | values (tok/s) |
|---|---|---|---|---|
| none | 1 | **1.016x** | 4.6h | 26.3, 26.7, 26.3 |
| none | 16 | **1.105x** | 2.7h | 244.2, 253.2, 269.9 |
| dflash | 1 | **1.663x** | 4.2h | 28.4, 33.2, 47.3 |
| dflash | 16 | **1.624x** | 2.3h | 436.0, 356.5, 578.9 |

The non-speculative rows are the control, and they are a good one: same
container lifecycle, same host, same time gaps, same prompts, differing only
in whether speculation is on. They move 1.02-1.11x. The speculative rows
move 1.62-1.66x. Whatever varies is specific to the speculative path rather
than a property of the machine or the day.

The c=16 figure is the sharpest part. This data gives **1.624x**; the
controlled concurrency test above, run separately and deliberately, gives
**1.64x** for the same cell. Two independent measurements agreeing to within
1%, one of them recorded before anyone was looking for the effect, so it
cannot be an artifact of how the test was designed.

This also means the effect was present and visible in the grid the whole
time. It went unnoticed because `analysis/perf_tables.py` averaged those
three measurements into one mean and described the result as prompt
variance within a single boot -- a provenance claim the table asserted and
never checked. It now checks it.

## Provenance gap (recorded 2026-09-23)

The measurements behind this finding were taken by a bespoke GPU probe that
wrote **nothing to `results/`**. Its only output was stdout, so the numbers
above currently rest on terminal scrollback -- which `docs/data-model.md`
section 4 rule 4 explicitly forbids, and which means a reader cannot check
them against raw records.

This is not a claim that the numbers are wrong; it is a claim that they are
presently **unverifiable by a third party**, which for this study's purposes
is close to the same thing.

The probe now writes one record per measurement to `results/probes.jsonl`
and persists each boot's engine log, so a re-run produces citable evidence.
`analysis/check_provenance.py` fails on this finding until it does.

For this finding specifically, `sweeps/perf_v2.yaml` supplies the evidence as
a side effect of its main purpose: it carries `enforce_eager` as an axis
across all eight cells and runs through `specfp8.sweep`, which records
normally. That is stronger than re-running the probe, since it tests the
claim on eight configurations rather than the one this finding used.

## The cleanest instance: one cell, both modes, same argv

The perf_v2 grid caught cell `5824814b68901229`
(`dflash | bf16 | fp8_e4m3 | FLASHINFER`, **graphs on**) in the other mode.
Same `cell_id`, therefore same configuration by construction, and the
`argv_fingerprint` matches:

| session | c=1 tok/s | mean | tau |
|---|---|---|---|
| 2026-09-22 (perf) | 28.59, 29.39, 33.81 | **30.60** | 4.183 |
| 2026-09-22 (perf_v2) | 64.89, 71.77 | **68.33** | 4.211 |

**2.23x throughput. tau differs by 0.68%, below the 1.32% floor (F006).**

Every earlier demonstration of the bimodality compared *something* —
different arms, different cells, or boots inside a deliberately constructed
test. This is a single content-addressed configuration, measured in two
ordinary grid runs weeks apart in intent and hours apart in fact, landing in
both modes with acceptance unmoved. Nothing was varied, because there was
nothing left to vary: the `cell_id` is the configuration.

It also falsifies a premise of the pre-registered prediction in
`docs/paper-outline.md`, which assumed bf16 speculative cells sit near 30
tok/s with graphs on and would rise to 70-80 only with graphs off. Tonight
that cell is at 68.33 **with graphs on**. The premise was not wrong so much
as *not a property of the configuration*: which mode a graphs-on boot lands
in is exactly what is unpredictable, so any prediction conditioned on it is
conditioned on a coin flip. That is the finding, restated as a cost.

## Major correction: the speedup is not a property of eager

`perf_v2` measured both arms of `dflash | bf16 | auto` **within one run**,
and the graphs-on boot landed in the **fast** mode:

| conc | graphs on | graphs off | off/on |
|---|---|---|---|
| 1 | 70.0 | 79.5 | **1.13x** |
| 16 | 739.4 | 781.9 | **1.06x** |
| 64 | 1534.9 | 1605.2 | **1.05x** |

This finding reports **2.38x / 1.67x / 1.37x** for the same configuration.
The difference is not measurement error. It is which mode the *comparison
arm* drew: both default boots in the concurrency test landed slow, and
tonight's landed fast. Against the prior session's slow-mode boot the same
eager numbers give 2.19x at c=1.

**So the ratio is not a constant and should never have been reported as
one.** The defensible statement is:

- **graphs off** is stable at ~79.5 tok/s at c=1 (boot spread 1.04-1.07x).
- **graphs on** is bimodal: ~30 tok/s or ~70 tok/s.
- Eager's advantage is therefore **~1.05-1.13x against a fast-mode boot**
  and **~2.2x against a slow-mode one**. A single number for it is a
  statement about a coin flip, not about the flag.

The throughput headline of this finding is withdrawn and replaced by the
above. What survives — and is strengthened — is the **stability** claim:
eager removes a 1.64x boot-to-boot spread and replaces it with 1.04-1.07x.
That was always the more useful half, and it is now the whole of it.

This is the study's own thesis turned on itself. The 2.38x was produced by
comparing against an arm assumed representative, which was in fact one draw
from a bimodal distribution this very finding describes. Recorded as F022
instance 15.

## Qualified by F023: eager is not free

`--enforce-eager` increases KV allocation rather than reducing it. vLLM's
memory profiler reserves headroom for CUDA-graph capture, so graphs-on runs
at an effective utilization of 0.8843 instead of 0.92; removing graphs hands
that reservation to the KV cache. On a 22 GiB L4 that is enough to make
`dflash | bf16 | fp8_e4m3 | eager` fail to boot outright (F023).

So the recommendation here holds only for configurations that can boot with
it. The honest form is: eager removes the bimodality and is faster at every
concurrency tested, **on cells where it starts**, and it costs headroom that
memory-tight hardware may not have.

## What this does NOT establish

- **Why a default boot picks one mode over the other.** The selector is
  still unidentified, and remains the open question. What is now clear is
  that the selection is made once per boot and then holds.
- **That the effect generalises** beyond dflash / FLASHINFER / SM89 /
  vLLM 0.29.0.
- **That pinning eager for the grid is fair.** `--enforce-eager` is
  engine-wide. Its effect was measured only on speculative cells, where the
  spec-decode fallback exists; on the non-speculative baseline CUDA graphs
  have no fallback to trip over and are expected to help. Applying eager to
  both arms of a comparison whose baseline it penalises would manufacture
  part of the speculative speedup. Measured separately at `mechanism=none`
  before the grid is re-run — see the Consequence section.

## Consequence

The advantage holds at every concurrency, so the speculative half of the
Phase 2 grid can be re-measured with eager pinned at **one boot per cell** —
cheaper than the original sweep, and measuring a configuration that is
stable by construction rather than averaging over a bimodal one.

This also retires a defect in `sweeps/perf.yaml` that nothing had caught.
Its `repeat: [0, 1, 2]` runs three times *within one boot* (sweep.py boots
once and runs many), so the repeats vary prompts but never vary the boot —
they never sampled the dominant variance source at all. Under the default
arm those three numbers would have looked tight while sitting somewhere
inside a 1.64x boot-level spread. Under eager the boot term collapses to
1.04-1.07x and within-boot repeats become the right granularity again.

One thing is still open before the pin is applied: whether eager costs the
**non-speculative** baseline anything. That is the fairness question above,
and it decides the grid design:

- **Eager neutral or better at `mechanism=none`** — pin it for all cells.
  One controlled execution path, one boot per cell.
- **Eager worse at `mechanism=none`** — pinning it for both arms would
  inflate the speculative speedup. The grid then reports each arm at its
  own best setting, and says so explicitly, rather than holding a knob
  fixed at a value that is only right for one side.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine bootvar --repeats 2
```

```bash
modal run cloud/modal_probe.py --engine bootvar --sweep none --repeats 2
```
