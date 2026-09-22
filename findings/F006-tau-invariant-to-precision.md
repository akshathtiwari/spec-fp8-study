---
id: F006
title: tau is invariant to FP8 precision; backend choice perturbs it more
kind: result
status: established
confidence: medium
superseded_by: []
date: 2026-09-21
evidence:
  runs: [2026-09-21T04-32Z_h1, 2026-09-21T07-41Z_backend]
  analysis: [analysis/out/tables/compatibility_matrix.md]
  code_sha: b294050
---

## Withdrawal rescinded — see F019

An earlier notice here withdrew part of this finding on the basis of F018,
which claimed `--attention-backend` was ignored. F018 was wrong: the flag
controls the target model, and F018's parser had read the draft's independent
selection. This finding stands as written. A speculative cell does involve two
backends, so read it alongside F019.

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

- **tau values are tied to the prompt set that produced them.** These were
  measured over the pre-2026-09-21 gate prompts, whose GSM8K half was later
  found to be corrupted and replaced (F013). Re-measuring the identical
  configuration `dflash|bf16|auto|FLASHINFER` over the corrected prompts gives
  tau 2.77 against 2.716 here — a 2% shift from the workload alone. Acceptance
  depends on what is being generated, so absolute tau must not be compared
  across tables built on different prompt sets. The *comparison* in this
  finding is unaffected, because every cell in it was measured over the same
  set; only the absolute values move.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine vllm --sweep h1
modal run cloud/modal_probe.py --engine vllm --sweep backend
```
