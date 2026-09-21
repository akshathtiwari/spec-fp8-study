---
id: F004
title: FP8 KV cache doubles measured KV capacity when backend is held fixed
kind: result
status: established
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T07-41Z_backend]
  cells: [c6519fb025e64635, 5824814b68901229, 154965a1bfbfe3af, 02f47ab1d7e7f6b9]
  analysis: [analysis/out/tables/kv_capacity.md]
  code_sha: b294050
---

## Claim

Holding model, mechanism, weights and **attention backend** fixed, switching the
KV cache from BF16 to `fp8_e4m3` multiplies measured KV token capacity by
**1.994x – 2.000x** (n=4), against a theoretical 2.000x for 16-bit to 8-bit.

## Evidence

| mech | backend | BF16 KV | FP8 KV | ratio |
|---|---|---|---|---|
| dflash | FLASHINFER | 58,416 | 116,832 | 2.000x |
| dflash | TRITON_ATTN | 58,528 | 116,832 | 1.996x |
| none | FLASHINFER | 73,888 | 147,776 | 2.000x |
| none | TRITON_ATTN | 77,168 | 153,872 | 1.994x |

Capacity is parsed from the engine's own `GPU KV cache size: N tokens` line,
i.e. measured rather than computed from a formula (R8).

## Reasoning

The ratio landing on the theoretical maximum indicates the saving is the raw
element-width change with negligible added overhead. Separately, DFlash costs
capacity: at BF16 on FLASHINFER, drafting reduces capacity from 73,888 to 58,416
(0.79x) because the draft model occupies memory the cache would otherwise use.
FP8 KV more than repays that.

## What this does NOT establish

That the extra capacity converts into throughput. Capacity is a precondition for
concurrency, not a measurement of it; the goodput sweep (R6/R7, Increment 2) is
unbuilt, so no serving-rate claim follows from this yet.

## How to reproduce

```bash
python analysis/build_tables.py   # regenerates analysis/out/tables/kv_capacity.md
```
