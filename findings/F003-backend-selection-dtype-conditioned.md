---
id: F003
title: vLLM's attention backend selection is conditioned on KV cache dtype
kind: result
status: established
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T07-41Z_backend]
  cells: [6d744b9cd1cdd0ca, 66da0689042ca795, 20224e954c5db2f6, eca0867b82c947ad]
  analysis: [analysis/out/tables/backend_attribution.md]
  code_sha: b294050
---

## Claim

With `attn_backend: auto` on SM89, vLLM 0.29.0 selects **FlashAttention-2** at
BF16 KV and switches to **FlashInfer** when `fp8_e4m3` KV is requested. The
backend is therefore not a free variable a user sets, but a function of the
precision they ask for.

## Evidence

From the engine's startup log (`results/logs/6d744b9cd1cdd0ca.log`):

```
[cuda.py:492] Using FLASH_ATTN attention backend out of potential backends:
              ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
[flash_attn.py:897] Using FlashAttention version 2
```

| cell | mech | kv | requested | actual |
|---|---|---|---|---|
| `6d744b9c` | dflash | auto | auto | FLASH_ATTN |
| `66da0689` | dflash | fp8_e4m3 | auto | FLASHINFER |
| `20224e95` | none | auto | auto | FLASH_ATTN |
| `eca0867b` | none | fp8_e4m3 | auto | FLASHINFER |

## Reasoning

The candidate set on SM89 is four backends. Requesting FP8 KV moves the
selection from FLASH_ATTN to FLASHINFER for both the causal and non-causal
drafting paths, so the trigger is the KV dtype rather than the mechanism.

Two consequences matter for the study. First, the earlier `h1.yaml` sweep did
exercise FlashInfer after all, because its `fp8_e4m3` cells were auto-selected
onto it — the concern that H1's mechanism had never been tested was unfounded.
Second, any comparison that varies KV precision while leaving the backend on
`auto` silently varies the backend too (see F005).

## What this does NOT establish

Why the selection policy is written this way, or whether it holds on other
compute capabilities. The candidate list is SM89-specific.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine which-backend
```
