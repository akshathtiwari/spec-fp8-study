---
id: F016
title: FP8 weights and FP8 KV cost no measurable task accuracy, within +/-4.7 points
kind: result
status: established
confidence: medium
date: 2026-09-21
evidence:
  runs: [2026-09-21T11-2xZ_quality]
  cells: [aaeb804d4d2a584f, fa7996fe5764f060, cc2e011a0a75bf57, f4d8fbdfdccc68f6, c6519fb025e64635, 5824814b68901229, b6d238cffb8de674, 551ab2aa2e3b1cc5]
  analysis: [analysis/out/tables/quality.md]
  code_sha: 1085032
supersedes: []
---

## Claim

On Qwen3-4B over 256 GSM8K problems, quantizing weights to FP8, the KV cache to
`fp8_e4m3`, or both, produces no detectable change in task accuracy. Every
paired comparison bounds the difference within **+/-3.0 to +/-4.7 percentage
points** with point estimates near zero, for both non-speculative decoding and
DFlash block drafting.

## Evidence

Modal L4 (SM89), vLLM 0.29.0, `attn_backend` pinned to FLASHINFER, thinking
disabled, 768-token budget, scored from stored generations.

| mechanism | weights | kv_cache | accuracy | 95% CI | unparseable |
|---|---|---|---|---|---|
| none | bf16 | auto | 85.9% | 81.1-89.7 | 0 |
| none | bf16 | fp8_e4m3 | 87.5% | 82.9-91.0 | 0 |
| none | fp8 | auto | 88.3% | 83.8-91.7 | 0 |
| none | fp8 | fp8_e4m3 | 87.1% | 82.4-90.7 | 0 |
| dflash | bf16 | auto | 86.3% | 81.6-90.0 | 0 |
| dflash | bf16 | fp8_e4m3 | 87.9% | 83.3-91.3 | 0 |
| dflash | fp8 | auto | 86.3% | 81.6-90.0 | 0 |
| dflash | fp8 | fp8_e4m3 | 87.1% | 82.4-90.7 | 0 |

Paired (McNemar, exact binomial) across all 12 within-mechanism comparisons:
discordant pairs 16-23 of ~250, p from 0.238 to 1.000, and every 95% interval
on the difference contained within +/-4.7 points.

## Reasoning

The configurations agree on roughly 92% of problems; the ~8% they disagree on
split about evenly, so there is no directional damage.

**The interval is the result, not the p-value.** With ~20 discordant pairs,
reaching p<0.05 would need a nearly one-sided split, so "not significant" would
be equally true of no effect and of a fairly large one. The interval states
what is excluded: an effect larger than about 4 points is ruled out; anything
smaller is not resolved.

The backend is pinned rather than left on `auto` deliberately. vLLM serves
BF16 KV on FlashAttention-2 and FP8 KV on FlashInfer (F003), so under `auto` a
bf16-vs-fp8 comparison changes backend and precision together and can attribute
a difference to neither.

Read with F006 (tau unchanged under FP8) and F004 (FP8 KV roughly doubles KV
capacity), the practical statement is that on Ada, FP8 KV buys ~2x cache
capacity at no cost in either acceptance or answer quality at this resolution.

## What this does NOT establish

- **One task, one model.** GSM8K arithmetic on Qwen3-4B. Says nothing about
  long-context, code, multilingual or structured-output tasks, where
  quantization error may accumulate differently.
- **Effects below ~4 points are unresolved**, and a 3-point accuracy loss could
  matter in production. Narrowing this needs more problems, not more cells.
- **Greedy decoding only**, temperature 0, thinking disabled.
- **Not a claim of equivalence.** Absence of a detected difference at this
  power is not evidence of no difference.

## Provenance caution

Four defects had to be fixed before these numbers meant anything, and each
produced plausible output rather than an error: corrupted reference answers
(F013), a 256-token budget truncating reasoning (F014), an extractor
discarding 58.6% of correct answers (F015), and scoring that had to be moved
into the analysis layer so it could be corrected without re-running hardware.
Accuracies reported before 2026-09-21 are superseded and must not be cited.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine prefetch --sweep quality
modal run cloud/modal_probe.py --engine vllm --sweep quality --quality
python analysis/build_tables.py
```
