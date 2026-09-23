---
id: F026
title: Client-measured inter-token latency is inter-chunk latency, so an ITL-based SLO penalises speculative decoding by a factor of tau and inverts the comparison it is meant to make
kind: result
status: established
confidence: high
date: 2026-09-23
evidence:
  runs: [2026-09-23T05-0xZ_perf_v2]
  analysis: [analysis/out/tables/goodput.csv, analysis/out/tables/goodput.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

A streaming client measures the gap between **arrivals**, and under
speculative decoding one arrival carries several accepted tokens. The
quantity everyone calls "inter-token latency" is therefore
**inter-chunk latency**, and on a speculative server it is larger than the
true per-token interval by roughly **tau**.

An SLO expressed as "p95 ITL <= X ms", computed from a stream the way every
load generator computes it, therefore **penalises speculative decoding by
about 4x on this configuration** — and reports the slower system as the
compliant one.

## Evidence

Grid measurements at concurrency 64, CUDA graphs on, FP8 weights, auto KV,
identical prompts and SLO:

| | measured gap | tokens per chunk | implied per-token | raw throughput |
|---|---|---|---|---|
| `dflash` (speculative) | **93.8 ms** | **4.122** | **22.8 ms** | **2219 tok/s** |
| `none` | 33.8 ms | 1.007 | 33.6 ms | 1281 tok/s |

Speculation is **1.5x faster per token** and appears **2.8x slower** on the
measured metric.

The consequence in the goodput table, at p95 ITL <= 50 ms and
TTFT <= 1000 ms:

| cell | conc | raw req/s | met | goodput req/s |
|---|---|---|---|---|
| dflash \| fp8 \| auto | 64 | **8.31** | **0%** | **0.000** |
| dflash \| bf16 \| auto | 64 | 6.13 | 0% | 0.000 |
| none \| fp8 \| auto | 64 | 5.20 | 100% | **5.188** |
| none \| bf16 \| auto | 64 | 3.37 | 6% | 0.209 |

Every speculative configuration scores **zero goodput** while having the
highest raw throughput in the grid. The fastest system in tokens per second
is ranked last by the SLO.

At concurrency 1 the effect is present but does not bite: chunk gaps are
small in absolute terms, so speculative cells meet the same target 100% of
the time. The inversion appears exactly where serving actually operates.

## Reasoning

vLLM streams one SSE chunk per *decoding step*. Without speculation a step
emits one token, so arrivals and tokens coincide and `tokens_per_chunk` is
1.007 — the small excess being multi-byte characters split across chunks.
With speculation a verification step emits every accepted draft token at
once, so `tokens_per_chunk` is 4.122, close to the measured tau of ~4.2.

This is not a vLLM defect and not a measurement bug in this harness. It is
a property of the interface: the server has no way to deliver four accepted
tokens at four different times, because it produced them at one time. Any
client measuring arrival gaps on any engine will see the same thing.

It matters because ITL is normally used as a **smoothness** proxy — "does
the text appear at a comfortable rate" — and under speculation it stops
measuring that. Text arriving in bursts of four every 94 ms is not
equivalent to text arriving singly every 94 ms, but it is also not
equivalent to text arriving singly every 23 ms. Which one a user perceives
is a UX question this study cannot answer.

## What this does NOT establish

- **The right metric.** Dividing the gap by `tokens_per_chunk` recovers a
  sensible per-token figure, but that is an average and discards the
  burstiness that a smoothness SLO exists to capture. This finding shows
  the current metric is wrong under speculation; it does not claim the
  quotient is right.
- **That perceived smoothness follows either number.** No user study was
  run. Burst-of-4-every-94ms may well read as smooth at these rates.
- **That tau is the exact factor.** `tokens_per_chunk` is 4.122 and tau is
  ~4.2; they track but are not identical, since a chunk boundary is not
  exactly a verification step when multi-byte characters split.
- **Anything about non-streaming serving.** A batch API has no arrival
  gaps and no such distortion.

## Consequence

**For this paper.** This is a fourth confound, and structurally the same as
the other three: a number that looks like a measurement of the thing under
study and is a measurement of something else. It belongs beside
dtype-conditioned backend selection, the CUDA-graph bimodality, and the
acceptance noise floor.

It is also the most consequential of the four for practitioners, because it
does not merely add noise — it **reverses the ranking**. A team evaluating
speculative decoding against an ITL SLO, using a standard load generator,
would conclude speculation made their service non-compliant while it was in
fact producing 1.7x the tokens.

**For the goodput results.** Speculative rows at concurrency 64 in
`goodput.md` should be read as "meets an inter-chunk SLO 0% of the time",
not as "fails to serve". Reporting them without this finding attached would
be actively misleading, which is why the paper reports both.

**For prior art.** The related work in `requirements.md` §9 reports
single-request latency, where the effect is present but does not bite. That
is consistent with nobody having reported it.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine sweep --sweep perf_v2 --force
```

```bash
python analysis/goodput.py
```

The per-request rows in `results/runs/` carry `itl_ms` and the summary
records carry `tokens_per_chunk`; the ratio between a speculative and a
non-speculative cell at the same concurrency is the whole finding.
