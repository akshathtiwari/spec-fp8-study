---
id: F006
title: tau is invariant to FP8 precision; backend choice perturbs it more
kind: result
status: established
confidence: medium
date: 2026-09-21
evidence:
  runs: [2026-09-21T04-32Z_h1, 2026-09-21T07-41Z_backend]
  analysis: [analysis/out/tables/compatibility_matrix.md]
  code_sha: b294050
---

## Claim

For DFlash on Qwen3-4B, acceptance length is unaffected by FP8 quantization of
either weights or KV cache, and is perturbed slightly more by the choice of
attention backend than by precision.

## Evidence

Precision sweep at `attn_backend: auto` (run `2026-09-21T04-32Z_h1`):

| weights | kv | tau | acceptance |
|---|---|---|---|
| bf16 | auto | 2.685 | 0.3371 |
| bf16 | fp8_e4m3 | 2.699 | 0.3398 |
| fp8 | auto | 2.690 | 0.3380 |
| fp8 | fp8_e4m3 | 2.680 | 0.3361 |

Range 2.680–2.699 = **0.7%**.

Backend sweep at bf16 weights (run `2026-09-21T07-41Z_backend`):

| kv | auto (FA2) | FLASHINFER | TRITON_ATTN |
|---|---|---|---|
| bf16 | 2.703 | 2.716 | 2.651 |
| fp8_e4m3 | 2.695 | 2.697 | 2.745 |

Range 2.651–2.745 = **3.6%**.

## Reasoning

The FP8 column is not systematically below the BF16 column — on TRITON_ATTN it
is higher — so there is no directional degradation to attribute to
quantization. That FP8 does not reduce acceptance is not obvious a priori: the
draft model was trained against an unquantized target, so quantizing the target
could plausibly have shifted its outputs away from what the draft predicts.

The backend spread being larger is worth stating because it constrains method:
tau comparisons are only meaningful with the backend held fixed, and F003 shows
`auto` does not hold it fixed across a precision change.

## What this does NOT establish

- **One model, one drafter, one depth.** Qwen3-4B with
  `z-lab/Qwen3-4B-DFlash-b16` at 5 speculative tokens. ngram was measured only
  on Qwen3-0.6B (tau 1.473) and eagle3/mtp not at all.
- **No dispersion.** Each tau is a single measurement over 32 prompts with no
  repeats, so the reported spreads have no error bars and R10 (minimum 3
  repeats) is unmet. The differences discussed are small enough that repeats
  could change the ordering.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine vllm --sweep h1
modal run cloud/modal_probe.py --engine vllm --sweep backend
```
