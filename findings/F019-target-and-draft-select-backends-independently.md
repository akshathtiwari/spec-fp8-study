---
id: F019
title: Target and draft select attention backends independently; F018 was wrong
kind: result
status: established
confidence: high
date: 2026-09-22
evidence:
  runs: [2026-09-21T07-41Z_backend, 2026-09-21T14-5xZ_backend_gaps, 2026-09-21T17-08Z_repeat]
  cells: [154965a1bfbfe3af, 02f47ab1d7e7f6b9, 9e05fa1354abf37b]
  analysis: [analysis/out/tables/backend_attribution.md]
  code_sha: 007d5ed
supersedes: [F018]
---

## Claim

In speculative decoding, vLLM 0.29.0 selects an attention backend **separately
for the target and for the draft**. `--attention-backend` controls the
**target** and is honoured. The **draft auto-selects**, ignoring the flag, and
its choice is conditioned on KV dtype: FLASH_ATTN at BF16, FLASHINFER at
fp8_e4m3.

**This retracts F018 entirely.** The flag was never ignored. F018 read only the
draft's selection line and concluded the target's request had been discarded.

## Evidence

A DFlash cell requesting `TRITON_ATTN` logs two selections, and the checkpoint
size after each identifies which model it belongs to:

```
[model_runner.py:382] Loading model from scratch...
[cuda.py:432] Using AttentionBackendEnum.TRITON_ATTN backend.      <- target
[weight_utils.py:863] Checkpoint size: 7.49 GiB                     <- Qwen3-4B
...
[vllm.py:1924] max_num_scheduled_tokens ... based on the speculative decoding settings
[cuda.py:492] Using FLASH_ATTN attention backend out of potential
              backends: ['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION']
[weight_utils.py:897] Using FlashAttention version 2
[weight_utils.py:863] Checkpoint size: 1.00 GiB                     <- DFlash draft
```

7.49 GiB and 1.00 GiB match the two checkpoints exactly as prefetched
(Qwen3-4B 7.51 GiB, `z-lab/Qwen3-4B-DFlash-b16` 1.00 GiB).

Across every bf16 cell, target vs draft:

| mechanism | requested | KV | target | draft |
|---|---|---|---|---|
| dflash | FLASHINFER | bf16 | FLASHINFER | FLASH_ATTN |
| dflash | FLASH_ATTN | bf16 | FLASH_ATTN | FLASH_ATTN |
| dflash | FLEX_ATTENTION | bf16 | FLEX_ATTENTION | FLASH_ATTN |
| dflash | TRITON_ATTN | bf16 | TRITON_ATTN | FLASH_ATTN |
| dflash | TRITON_ATTN | fp8_e4m3 | TRITON_ATTN | FLASHINFER |
| none | * | * | as requested | (single model) |

Non-speculative cells log one selection. Two lines appear only when there is a
draft.

## Why F018 got it wrong

The two selections use different wording because they take different code
paths: `cuda.py:432` `"Using AttentionBackendEnum.X backend."` when forced,
`cuda.py:492` `"Using X attention backend out of potential backends: [...]"`
when auto-selecting. F018's parser matched only the second form, so in a DFlash
cell it read the **draft's** auto-selection and reported it as the cell's
backend — making an honoured request look discarded.

The parser now tries the forced form first and falls back to the auto form.

## What this restores

- **F017 stands.** `TRITON_ATTN` genuinely serves fp8_e4m3, verified twice:
  `none|fp8_e4m3|TRITON_ATTN` (single model, target TRITON_ATTN, ok) and
  `dflash|fp8_e4m3|TRITON_ATTN` (target TRITON_ATTN, ok). The FLASH_ATTN and
  FLEX_ATTENTION rejections were never in doubt.
- **F006's backend observation stands.** Target backend does move tau:
  FLASHINFER 2.708, FLEX_ATTENTION 2.719, FLASH_ATTN 2.740, TRITON_ATTN 2.769,
  all at bf16 with the draft on FLASH_ATTN. Spread 0.061 against a repeat
  reproducibility of <= 0.002, so it is a real effect.
- **F004's backend labels are valid** — they are target backends.

## What it adds

A speculative cell is **two backend choices, not one**, and only one of them is
controllable. Any claim of the form "mechanism X on backend Y" is incomplete
for a speculative configuration unless it names both. The draft's choice also
shifts with KV dtype, so changing precision changes the draft backend even when
the target is pinned — a confound that applies to every precision comparison
run with speculation.

## What this does NOT establish

Whether the draft's backend can be controlled at all. `--attention-config` was
only exercised on a non-speculative cell before the sweep was stopped, so
whether it reaches the draft is untested.

## Consequence for the public record

A comment on vllm-project/vllm#44879 was posted retracting the Triton claim on
F018's basis. That retraction was itself wrong and has been corrected again.
