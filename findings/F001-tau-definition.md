---
id: F001
title: tau must add the bonus token; the engine counter excludes it
kind: method
status: established
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T04-32Z_h1, 2026-09-21T07-41Z_backend]
  cells: [6d744b9cd1cdd0ca]
  analysis: [analysis/out/tables/compatibility_matrix.md]
  code_sha: b294050
supersedes: []
superseded_by: []
---

## Claim

Mean accepted length is `tau = 1 + accepted_draft_tokens / verification_steps`,
not `accepted / steps`. The study's headline metric was originally defined the
second way and returned values below 1.0, which is impossible for a count of
tokens emitted per step.

## Evidence

vLLM 0.29.0 exposes `vllm:spec_decode_num_accepted_tokens_total`,
`vllm:spec_decode_num_draft_tokens_total` and
`vllm:spec_decode_num_drafts_total`. An observed run recorded 650 drafts at 3
speculative tokens each (1950 draft tokens) with 185 accepted.

## Reasoning

185 accepted is far below the 650 floor that would apply if the counter included
the bonus token, since every verification step emits the target's own token
whether or not any draft survives. The counter therefore counts accepted *draft*
tokens only, and the bonus must be added back. This also bounds tau below by 1.0
(speculation contributing nothing) and above by 1 + num_speculative_tokens.

`acceptance_rate = accepted / draft_tokens` is reported separately because it
measures draft quality independently of speculation depth, so it stays
comparable across mechanisms configured with different depths.

## What this does NOT establish

Nothing about SGLang's counters, which have their own names and may or may not
adopt the same bonus-token convention. `metrics/sglang.py` is unexercised.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine diagnose
```
prints the raw counters before and after a load, from a live server.
