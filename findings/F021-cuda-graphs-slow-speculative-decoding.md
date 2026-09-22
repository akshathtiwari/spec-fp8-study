---
id: F021
title: Speculative throughput is bimodal, and enforce_eager reliably selects the fast mode
kind: result
status: provisional
confidence: medium
date: 2026-09-22
evidence:
  runs: [2026-09-22T10-0xZ_boot_variance, 2026-09-22T09-07Z_boot_stability]
  analysis: []
  code_sha: 624ffdb
supersedes: []
---

## Claim

The 2.16x spread in speculative throughput (findings/F020) is **not
continuous variance**. It is **bimodal**: boots land in either a slow mode
near 30 tok/s or a fast mode near 58-61 tok/s. Disabling CUDA graphs with
`--enforce-eager` reliably selects the fast mode, making speculative decoding
**~1.9x faster** at identical acceptance.

## Evidence

Three arms, three boots each, one container, identical prompts, dflash /
bf16 / auto KV / FLASHINFER / gsm8k c=1 n=16:

| arm | mean tok/s | s.d. | min - max | ratio | tau |
|---|---|---|---|---|---|
| default | 30.57 | 4.3% | 29.07 - 31.50 | 1.08x | 4.09-4.12 |
| **enforce_eager** | **58.33** | 5.9% | 54.37 - 60.65 | 1.12x | 4.143 |
| piecewise (pinned) | 30.04 | 6.0% | 28.09 - 31.69 | 1.13x | 4.077 |

Re-reading the six boots behind F020 in this light, they cluster rather than
scatter: **28.4, 33.2** against **50.7, 60.8, 61.2**. Two groups, not a
spread.

## Reasoning

The hypothesis in F020 — that CUDA-graph *capture* varies per boot — is
**wrong**. Every arm here is individually stable (1.08-1.13x), including the
default arm that produced both modes across earlier runs. Whatever selects
the mode is fixed early and then holds.

What the test does establish is the direction: turning CUDA graphs **off**
nearly doubles speculative throughput. That is consistent with the warning
the engine emits at every speculative boot —

```
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
attention backend FlashInfer
```

— followed by a fallback that evidently costs more than it saves here.
Pinning `cudagraph_mode: PIECEWISE` explicitly reproduces the slow mode
exactly (30.04 vs 30.57), which indicates the fallback *is* the slow mode
rather than being a third state.

tau is unchanged across all arms (4.08-4.14), so this is purely the cost of
producing accepted tokens, not how many are accepted — the same separation
F020 established.

## What this does NOT establish

- **Why a default boot picks one mode over the other.** All three default
  boots here landed slow; earlier boots of the same configuration landed
  fast. The selector is unidentified, and this is the open question.
- **That the effect survives under load.** CUDA graphs matter most at larger
  batch sizes, so the advantage could invert at concurrency 16 or 64. Tested
  only at concurrency 1, which is precisely the regime this study argues is
  misleading.
- **That it generalises** beyond dflash / FLASHINFER / SM89 / vLLM 0.29.0.

Status is `provisional` until the concurrency check is done.

## Consequence

If `enforce_eager` remains stable and fast under load, the speculative half
of the Phase 2 grid can be re-measured with it pinned at **one boot per
cell** — cheaper than the original sweep, and measuring a configuration that
is stable by construction rather than averaging over a bimodal one.

If it inverts under load, the grid needs boot-level repeats after all
(F020's Option A), and this becomes a low-concurrency-only result.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine bootvar --repeats 3
```
