---
id: F007
title: Exact-match equivalence testing saturates and cannot gate FP8 correctness
kind: method
status: established
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T04-32Z_h1, 2026-09-20T13-17Z_test_mini]
  analysis: [analysis/out/tables/correctness_tests.md]
  code_sha: b294050
---

## Claim

Design section 4 specifies Test 1 as "speculative vs non-speculative output at
matched precision, exact-match >= 0.95", and calls it the sharp instrument for
detecting a broken FP8 kernel. That threshold is unreachable on this stack: the
BF16 baseline is already 0.281, so the instrument has no dynamic range in which
to detect FP8 damage.

## Evidence

Test 1, dflash vs none at matched precision, Qwen3-4B:

| weights | kv | exact match | prefix agreement |
|---|---|---|---|
| bf16 | auto | **0.281** | 0.508 |
| bf16 | fp8_e4m3 | 0.156 | 0.438 |
| fp8 | fp8_e4m3 | 0.031 | 0.217 |

Test 2, non-speculative FP8 KV vs BF16 KV: exact match 0.125 at bf16 weights.

## Reasoning

Test 2 calibrates the confound. FP8 KV alone changes 87.5% of outputs on the
*causal* path, which is known-good, so most divergence is benign quantization
noise rather than a broken kernel. The mechanism is that quantization perturbs
logits, greedy argmax flips on near-ties, and a single flipped token diverges
everything after it — so over a 256-token generation, exact match measures
sequence length as much as correctness.

Test 1 falling under FP8 is therefore the expected consequence of a correct
implementation, not evidence against one. This is why F002 rests on tau instead:
tau degrades only if the drafts themselves become wrong, which is the property
actually at issue.

This generalises beyond this study: exact-match equivalence is the obvious
correctness test for quantized greedy decoding and it does not work at realistic
generation lengths.

## What this does NOT establish

That no FP8 damage exists — only that this instrument cannot detect it. A
replacement must be distributional or task-level.

## Related

Boot-to-boot determinism was verified separately (F009), so this saturation is a
property of the metric and not of measurement noise.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine correctness
```
