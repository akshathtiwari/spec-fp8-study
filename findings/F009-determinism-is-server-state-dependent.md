---
id: F009
title: Generation is exactly reproducible across boots but drifts within a server
kind: result
status: established
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-20T13-17Z_test_mini]
  analysis: []
  code_sha: b294050
---

## Claim

With temperature 0 on Qwen3-0.6B, two fresh servers with identical configuration
produce **32/32 = 100.0%** identical outputs, while a second pass against the
*same* server produces **29/32 = 90.6%**. Determinism is a function of server
state, not of the configuration alone.

## Evidence

`modal run cloud/modal_probe.py --engine determinism`, L4, BF16, non-speculative:

- A vs C (same config, fresh server): 32/32 = 100.0%
- A vs B (same server, repeated pass): 29/32 = 90.6%

## Reasoning

Prefix caching is the prime suspect: pass B reuses KV left by pass A, and a
cached prefix produces slightly different logits from a freshly computed one,
which flips greedy argmax on near-ties.

Two consequences, one reassuring and one not:

1. **The harness is sound for cross-cell comparison.** Every probe cell boots a
   fresh server and makes one pass, which is the 100% condition. Differences
   between cells are therefore attributable to configuration, which is what
   licenses F006 and F007 to compare cells at all.
2. **Server reuse is unsafe for correctness claims.** The design's Phase 2
   optimisation groups many RunCells onto one booted server, which is exactly
   the 90.6% condition.

## What this does NOT establish

That prefix caching is the cause — it is untested, the hypothesis rests on
mechanism rather than on an ablation. Disabling it and re-running would settle
it. Also only Qwen3-0.6B, BF16, non-speculative.

## Consequence

Recorded as T18b: Phase 2 must either re-boot per correctness arm or restrict
correctness claims to fresh-server cells. Throughput and latency measurements
are unaffected.
