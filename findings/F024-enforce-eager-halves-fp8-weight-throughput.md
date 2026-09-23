---
id: F024
title: enforce_eager costs FP8-weight configurations a fifth to a half of their throughput and costs BF16 nothing, so pinning it would have manufactured the speculative speedup on every FP8 cell
kind: result
status: established
confidence: high
date: 2026-09-23
evidence:
  runs: [2026-09-22T19-4xZ_perf_v2]
  analysis: [analysis/out/tables/cudagraph_arms.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

`--enforce-eager` costs a **large fraction of the throughput** of any
configuration using FP8 **weights** — between about a fifth and a half
across two independent grids — and costs a BF16-weight configuration
essentially nothing. The effect is present with and without speculation,
and with both KV dtypes.

The range is wide on purpose. The direction replicates on every cell; the
coefficient does not, because the eager+FP8 arm is itself unstable across
boots (see below). A single number here would be the same mistake F021 made
with its speedup ratio.

This is the reason `enforce_eager` had to be a grid axis rather than a
pinned setting. Pinning it — the obvious response to F021 — would have
halved the baseline of every FP8-weight cell while leaving BF16 alone, and
every speedup computed against those baselines would have been inflated by
about 2x.

## Evidence

`perf_v2`, 16 boots, both arms measured in the same run. Non-speculative
cells, individual boots rather than means, so the separation is visible:

| weights | KV | conc | graphs on | graphs off | off/on |
|---|---|---|---|---|---|
| bf16 | auto | 1 | 26.7, 27.1 | 25.6, 25.6 | **0.95x** |
| bf16 | auto | 64 | 776.8, 893.1 | 793.5, 801.5 | **0.96x** |
| bf16 | fp8_e4m3 | 1 | 26.4, 26.5 | 24.9, 25.9 | **0.96x** |
| bf16 | fp8_e4m3 | 64 | 804.3, 866.4 | 797.1, 883.2 | **1.01x** |
| **fp8** | auto | 1 | 41.1, 41.5 | **20.2, 20.5** | **0.49x** |
| **fp8** | auto | 16 | 403.1, 434.5 | 229.9, 238.8 | **0.56x** |
| **fp8** | auto | 64 | 1236.5, 1267.8 | 811.0, 839.9 | **0.66x** |
| **fp8** | fp8_e4m3 | 1 | 41.9, 42.6 | **21.2, 21.3** | **0.50x** |
| **fp8** | fp8_e4m3 | 16 | 448.9, 453.9 | 226.8, 239.0 | **0.52x** |
| **fp8** | fp8_e4m3 | 64 | 1370.9, 1380.6 | 750.5, 856.1 | **0.58x** |

The arms do not overlap on any FP8-weight row, and overlap on every BF16
row. Four configurations, three concurrencies, two repeats each.

Speculative cells show the same split:

| cell | conc | graphs on | graphs off | off/on |
|---|---|---|---|---|
| dflash \| bf16 \| auto | 1 | 70.0 | 79.5 | 1.13x |
| dflash \| bf16 \| auto | 64 | 1534.9 | 1605.2 | 1.05x |
| dflash \| **fp8** \| auto | 1 | 118.1 | 72.0 | **0.61x** |
| dflash \| **fp8** \| auto | 64 | 2065.2 | 1936.8 | **0.94x** |

tau is unaffected throughout (4.18-4.30 in both arms), so this is the cost
of producing tokens, not a change in acceptance.

## Replicated, with the magnitude not reproducing

The grid was run a second time end to end (16 boots, independent
containers). The **direction** replicates on every cell; the **size** does
not, and the reason is informative.

| cell | c | grid 1 | grid 2 | graphs-on g2/g1 | eager g2/g1 |
|---|---|---|---|---|---|
| none \| fp8 \| auto | 1 | 0.49x | 0.68x | 1.00x | **1.40x** |
| none \| fp8 \| auto | 64 | 0.66x | 0.78x | 1.03x | **1.22x** |
| none \| fp8 \| fp8_e4m3 | 1 | 0.50x | 0.66x | 0.99x | **1.31x** |
| none \| bf16 \| auto | 1 | 0.95x | 0.96x | 1.00x | 1.02x |
| none \| bf16 \| auto | 64 | 0.96x | 0.96x | 1.00x | 1.00x |
| dflash \| fp8 \| auto | 1 | 0.61x | 0.76x | 1.02x | **1.26x** |

Read the last two columns. The **graphs-on arm reproduces across grids on
every cell** (0.95-1.06x). The **eager arm reproduces for bf16 weights**
(1.00-1.05x) and **does not for FP8 weights** (1.22-1.40x).

So the ratio moved because the eager+FP8 arm is itself unstable across
boots, not because the baseline drifted. Within a boot it is tight — grid 1
measured 20.2 and 20.5 tok/s on its two repeats — and between boots it
moves 1.4x. That is boot-level instability, the same shape as F020, in a
different configuration.

**This study has now found two unstable dimensions, and each has a stable
partner.** Speculative decoding is bimodal with CUDA graphs *on* and stable
with them *off* (F021). FP8 weights are stable with graphs *on* and
unstable with them *off* (here). A practitioner cannot infer from one which
applies to the other, and neither can be discovered from a single boot.

The claim this finding makes is therefore directional and bounded, not a
coefficient: **eager costs FP8-weight configurations a large fraction of
their throughput — between about a fifth and a half in the measurements
here — and costs bf16 nothing.** Anyone wanting a coefficient needs
boot-level repeats, which is the same conclusion F020 reached for
speculation.

## Reasoning

The penalty tracks **weight precision**, not speculation and not KV dtype.
That points at the dequantisation path: an FP8 weight matmul on SM89 is
more kernel launches than a BF16 one, and CUDA graphs exist precisely to
amortise launch overhead. Removing them exposes it.

Consistent with that, the penalty **shrinks as concurrency rises** — 0.49x
at c=1 to 0.66x at c=64 — which is what per-launch overhead does when each
launch carries more work. It does not close, within the range measured.

That mechanism is a reading of the pattern, not something this study
measured. No kernel-level profiling was done, and the finding does not
depend on the explanation being right.

## What this does NOT establish

- **The mechanism.** See above: the dequantisation-launch story fits the
  shape of the data and is untested. A profile would settle it and was not
  run.
- **That it generalises off SM89 / vLLM 0.29.0.** FP8 weight handling is
  hardware- and version-specific, and Ada is not Hopper.
- **That FP8 weights are bad.** The opposite: with CUDA graphs on, FP8
  weights are the *fastest* configuration measured (1375 tok/s
  non-speculative at c=64 against 835 for bf16). The finding is about what
  disabling graphs costs them, not about FP8.
- **Anything about accuracy.** Unmeasured here; see F016.

## Consequence

Three things follow, in increasing order of importance.

1. **Practical.** Do not pair `--enforce-eager` with FP8 weights on this
   stack. It is the single most expensive flag combination in the grid.

2. **F021's recommendation is now narrow.** Eager removes the bimodality
   (F021), cannot boot speculative FP8-KV cells at all (F023), and costs
   FP8-weight configurations a fifth to a half of their throughput (here).
   It is defensible only for BF16-weight speculative configurations, where
   it buys stability at 1.05-1.32x throughput.

3. **Methodological, and the reason this finding exists.** The obvious
   response to F021 was to pin eager for the re-measured grid. Had that
   been done, every FP8-weight baseline would have been halved, every FP8
   speculative cell would have been halved or failed to boot, and the
   resulting speculation-vs-baseline speedups would have been inflated by
   roughly 2x on exactly the cells the paper is about — with nothing in the
   output to indicate it. The grid carried the flag as an axis specifically
   because that risk was identified in advance
   (`sweeps/perf_v2.yaml`, header), and the measurement confirms the risk
   was real and large.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine sweep --sweep perf_v2 --force
```

```bash
python analysis/cudagraph_arms.py
```
