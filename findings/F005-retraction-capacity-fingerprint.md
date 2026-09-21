---
id: F005
title: RETRACTED — KV capacity is not a reliable attention-backend fingerprint
kind: retraction
status: retracted
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T07-41Z_backend]
  analysis: [analysis/out/tables/kv_capacity.md, analysis/out/tables/backend_attribution.md]
  code_sha: b294050
supersedes: []
---

## Claim (withdrawn)

During the backend sweep it was proposed, mid-run, that measured KV capacity
could identify which attention backend served a cell, and that it be adopted as
a cheap cross-check in the paper's method section. **This is wrong and was
withdrawn.**

## What was believed, and why

The first three cells appeared to support it cleanly: `none` at BF16 KV gave
77,168 under `auto`, 73,888 under `FLASHINFER` and 77,168 under
`TRITON_ATTN`. The exact match between `auto` and `TRITON_ATTN`, with
FlashInfer differing, looked like a reliable discriminator, and it was reported
as showing `auto` selected Triton.

## Why it is wrong

The relationship does not survive further cells:

- At `fp8_e4m3` with `none`, `auto` and `FLASHINFER` both give 147,776
  while `TRITON_ATTN` gives 153,872 — the opposite association.
- At `fp8_e4m3` with `dflash`, `FLASHINFER` and `TRITON_ATTN` give the
  *same* 116,832 while `auto` differs at 115,920 — no discrimination at all.
- The original inference was also substantively wrong: the log shows `auto`
  selected **FLASH_ATTN**, neither of the two candidates being compared (F003).

Capacity reflects backend workspace reservation, which varies with dtype and
model configuration, so equal capacity does not imply equal backend.

## What caught it

Accumulating cells within the same sweep contradicted it before it reached any
written conclusion, and the persisted logs settled it definitively.

## Consequence

Backend attribution must come from `results/logs/`, which is what
`specfp8/analysis/backend_report.py` does. Capacity arithmetic is used only for
F004, where the backend is controlled explicitly rather than inferred.
