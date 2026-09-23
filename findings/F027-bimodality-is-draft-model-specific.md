---
id: F027
title: n-gram speculation shows no bimodality and no eager benefit across twelve boots, which localises the CUDA-graph effect to draft-model speculation rather than to speculative decoding as such
kind: result
status: established
confidence: high
date: 2026-09-23
evidence:
  runs: [2026-09-23T07-1xZ_boot_variance_ngram]
  analysis: [analysis/out/tables/boot_modes.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

F020 and F021 establish the bimodality entirely on DFlash. Repeating the
same protocol with **n-gram (prompt-lookup) speculation** finds **no
bimodality and no eager benefit**:

| mechanism | arm | boots | spread | tau spread | split? |
|---|---|---|---|---|---|
| ngram | default | **12** | **1.10x** | 1.011x | no |
| ngram | enforce_eager | 7 | **1.02x** | 1.000x | no |
| dflash | default | — | **~2.2x** | 1.007x | **yes** |

The effect is therefore associated with **draft-model-backed** speculation,
not with speculative decoding as such. That is a real narrowing of F020 and
F021, and it is the generalisation question a reader would ask first.

## Evidence

Two runs, **twelve** default-arm boots in total, identical prompts,
`bf16 | auto KV | FLASHINFER`, concurrency 1 / 16 / 64. Throughput at c=1,
sorted:

```
default (12)   32.35 34.02 34.15 34.16 34.71 34.89
               34.92 35.17 35.19 35.40 35.60 35.71    spread 1.10x
eager (7)      33.54 33.72 34.01 34.16 34.21 34.22 34.32   spread 1.02x
```

The twelve default boots are not merely unsplit, they are *tight*: the
entire range is 3.4 tok/s wide, narrower than the gap between two adjacent
DFlash boots inside the same mode.

Both arms are unimodal and indistinguishable from each other: mean 33.9
against 33.9, a ratio of **1.00x**. For DFlash the same comparison gives
1.05-1.32x, and the default arm splits into groups near 30 and near 65.

Two internal consistencies support the reading:

1. **No slow mode, no eager benefit.** F021's account is that eager helps
   by *avoiding* the slow mode. If ngram has no slow mode, eager should buy
   nothing — and it buys 1.00x. The absence of the benefit and the absence
   of the bimodality are the same observation, arriving independently.
2. **ngram acceptance is deterministic.** tau is `1.732` to three decimals
   on every default boot, a spread of 1.000x, whereas DFlash tau moves
   4.09-4.43 across boots. Prompt-lookup matching is a function of the text;
   a draft model's proposals depend on its own sampling. This is a sanity
   check that the harness is measuring what it claims: a deterministic
   mechanism reads as deterministic.

ngram's tau of 1.73 is far below DFlash's 4.2, as expected for prompt-lookup
on GSM8K, and its throughput at c=1 (33.9) sits between non-speculative
(~26) and DFlash's fast mode (~65).

## What this does NOT establish

- **That ngram is free of the effect** with certainty. Pooling the DFlash
  default-arm boots recorded across this study gives P(slow) near 0.38;
  twelve boots drawn from that distribution land all-fast **0.3%** of the
  time. That is strong enough to state the localisation without hedging and
  not strong enough to call the effect impossible in ngram.
- **That the draft *model* is the cause.** DFlash and ngram differ in more
  than one way: a separate model to load and schedule, a different
  acceptance distribution, tau 4.2 against 1.7, and different scheduler
  batch shapes. This localises the effect to the draft-model path; it does
  not isolate which property of that path matters. EAGLE-3 or MTP would
  narrow it further.
- **That tau being deterministic matters.** It is reported as a consistency
  check on the measurement, not as a mechanism.
- **Anything beyond SM89 / vLLM 0.29.0 / Qwen3-4B.**

## Consequence

**F020 and F021 must state the mechanism.** Their claims are about
speculative decoding *with a draft model* on this stack, not about
speculative decoding. The paper's §4 inherits the same narrowing, and the
honest version is stronger for being specific: an unqualified "speculative
throughput is bimodal" would have been refuted by the first reader who
tried n-gram.

**It also sharpens the practical advice.** A practitioner using
prompt-lookup speculation does not need boot-level repeats and gains
nothing from `--enforce-eager`. One using a draft model needs both, and on
FP8 weights the flag costs a fifth to a half of throughput (F024).

**And it reframes the study's contribution slightly.** The confound is not
"speculative decoding is unstable" but "the stability of a serving
configuration is not predictable from its parts". Two mechanisms that are
both speculative decoding, on the same engine, same card, same model,
behave oppositely under the same flag.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine bootvar --sweep ngram --repeats 8
```

```bash
python analysis/boot_modes.py
```
