---
id: F006
title: SUPERSEDED — the FP8 precision comparison is unreplicated and establishes neither difference nor equivalence
kind: result
status: superseded
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

> **SUPERSEDED 2026-09-24.** Both readings below are withdrawn. The
> precision comparison rests on **one boot per arm**, so it cannot establish
> a difference or an equivalence. The "1.32% noise floor" it invokes was the
> range of four boots, which is not a dispersion statistic (see the
> correction sections below and findings/F029 for the same error in another
> claim). The surviving measurement is that tau has CV 0.62% across 12
> fixed-prompt boots at tau~4.1; that number is not transferable to the
> operating point this finding measured.

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

## Cross-finding correction (2026-09-22): the backend claim is not separable from noise

F020 and F021 later measured tau's **boot-to-boot** variation for an identical
configuration with identical prompts: **1.32%** (4.0913 - 4.1452 over four
boots). Comparing that against this finding's two claims:

| claim | measured effect | boot-to-boot noise | separable? |
|---|---|---|---|
| precision does **not** move tau | 0.7% | 1.32% | effect is *below* noise — consistent with no effect |
| backend **does** move tau | 2.25% | 1.32% | only 1.7x noise, from **single boots** — not separable |

**The primary claim strengthens.** A precision effect of 0.7% sits below the
noise floor, so "tau is invariant to FP8 precision" is supported in the
precise sense that no effect larger than roughly 1.3% is detectable.

**The secondary claim weakens and is now withdrawn as stated.** The 2.708 -
2.769 spread across backends was measured one boot per backend, and at 1.7x
the boot-to-boot noise it cannot be distinguished from having sampled
different boots. Establishing it would need several boots per backend. This
is the third time this particular claim has moved — asserted, withdrawn by
F018 (wrongly), restored by F019, and now withdrawn on sounder grounds — and
the reason it kept moving is that it was always near the noise floor and
nobody had measured that floor.

Neither correction touches the precision comparison, which is what this
finding exists for.

### Correction to the correction (2026-09-24): the floor was transplanted

The comparison above is itself cross-era, and the reasoning has to be
restated.

The 1.32% floor was measured at **tau ~4.1**, on GSM8K with
`max_tokens=768`. The 2.25% backend spread was measured at **tau ~2.7**, on
the probe prompt set with `max_tokens=256`. Those are different operating
points, and this finding already noted in passing that workload alone shifts
tau by 2%. Comparing an effect at one operating point against a noise floor
measured at another assumes the *relative* floor is invariant across them,
which nobody checked.

**The withdrawal stands; the stated reason does not.** The defensible
argument needs no transplanted number: the backend spread was measured
**one boot per backend**, and F020/F021 establish that single boots are not
a sound basis for a throughput or timing claim on this stack. An unreplicated
2.25% difference is withdrawn because it is unreplicated, not because it
failed a comparison against a floor measured elsewhere.

Two consequences worth carrying:

1. **tau is not a fixed property of a draft/target pair.** It measured 2.71
   at `max_tokens=256` on probe prompts and 4.2 at `max_tokens=768` on
   GSM8K — a 55% difference from configuration alone. Any tau quoted without
   its generation length and prompt distribution is close to meaningless,
   and the literature quotes tau freely.
2. **The floor is operating-point-specific.** 1.32% is the floor *at tau
   ~4.1 on GSM8K at 768 tokens*. It should not be carried to other regimes,
   including by this finding.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine vllm --sweep h1
modal run cloud/modal_probe.py --engine vllm --sweep backend
```
