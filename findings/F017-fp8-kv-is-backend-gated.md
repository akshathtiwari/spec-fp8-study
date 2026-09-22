---
id: F017
title: FP8 KV works on exactly two of four SM89 backends, and vLLM's auto-switch is why
kind: result
status: superseded
confidence: high
superseded_by: [F018]
date: 2026-09-21
evidence:
  runs: [2026-09-21T14-5xZ_backend_gaps, 2026-09-21T07-41Z_backend]
  cells: [a59ad61dc4b05582]
  analysis: [analysis/out/tables/compatibility_matrix.md, analysis/out/tables/backend_attribution.md]
  code_sha: 017d655
---

## PARTIALLY WITHDRAWN — see F018

`--attention-backend` does not select a backend in vLLM 0.29.0; it is validated
but ignored. So the "ok" column below is not four backends, it is whichever
backend `auto` chose for that KV dtype.

**Still valid:** FLASH_ATTN and FLEX_ATTENTION genuinely *reject* fp8_e4m3, by
name, with the FA3/SM90 reason. The flag is honoured during validation.
**Withdrawn:** that TRITON_ATTN supports FP8 KV. That cell ran FlashInfer.

## Claim

Of the four attention backends vLLM 0.29.0 offers on SM89, **FP8 KV cache runs
on exactly two**: FLASHINFER and TRITON_ATTN. FLASH_ATTN and FLEX_ATTENTION
reject it at configuration time. This is the mechanism behind F003's
dtype-conditioned backend selection: `auto` does not switch away from
FlashAttention-2 under FP8 KV as a preference, it switches because
FlashAttention-2 **cannot** serve FP8 KV on this hardware.

## Evidence

Complete backend x KV-dtype matrix, bf16 weights, Qwen3-4B, L4 (SM89):

| backend | BF16 KV | fp8_e4m3 KV |
|---|---|---|
| FLASH_ATTN | ok (tau 2.74) | **rejected** |
| FLEX_ATTENTION | ok (tau 2.72) | **rejected** |
| FLASHINFER | ok (tau 2.72) | ok (tau 2.70) |
| TRITON_ATTN | ok (tau 2.65) | ok (tau 2.75) |

Verbatim, identical for both `none` and `dflash`:

```
ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not valid for
this configuration. Reason: ['kv_cache_dtype not supported',
'FP8 KV cache requires FA3 on SM90 or FA4 on SM100']

ValueError: Selected backend AttentionBackendEnum.FLEX_ATTENTION is not valid
for this configuration. Reason: ['kv_cache_dtype not supported']
```

## Reasoning

FlashAttention's message is the informative one: FP8 KV needs **FA3 on SM90 or
FA4 on SM100**, and Ada is neither. So the FlashAttention path genuinely is
Hopper-gated for FP8 KV, which is close to what H1 predicted — but H1 attached
the gate to FlashInfer, and on FlashInfer the combination works (F002). The
Hopper requirement is real; it simply lives in a different backend than the
hypothesis named.

This also **vindicates vLLM's behaviour rather than merely describing it**.
F003 recorded that `auto` serves BF16 KV on FlashAttention-2 and FP8 KV on
FlashInfer, which looked like an arbitrary quirk. It is not: FlashAttention-2
would fail, so selection must move. The same switch is what kept every earlier
`auto` cell working.

Note how these fail. Both are **clean configuration-time rejections with an
explicit reason**, not crashes and not silent corruption. That is the opposite
of H1's predicted failure mode — a configuration accepted by validation and
then failing at kernel dispatch. On 0.29.0 validation and capability agree.

The practical consequence for anyone deploying FP8 KV on Ada: two of the four
backends work, the default reaches a working one automatically, and pinning
`--attention-backend FLASH_ATTN` (a reasonable thing to do for speed at BF16)
breaks the moment FP8 KV is enabled.

## What this does NOT establish

- **One GPU, one version.** L4, vLLM 0.29.0, flashinfer 0.6.18.
- **Not tested on SM90.** The FA3 claim in the error message is vLLM's
  assertion, not something measured here; whether FA3 + FP8 KV works on Hopper
  is untested by this study.
- **Weights held at bf16** in the gap sweep, so FP8 weights x these backends is
  uncovered. There is no reason to expect it differs, but it was not measured.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine vllm --sweep backend_gaps
```
