---
id: F002
title: H1 refuted — FlashInfer serves FP8 KV with non-causal drafting on SM89
kind: result
status: established
confidence: medium
date: 2026-09-21
evidence:
  runs: [2026-09-21T07-41Z_backend]
  cells: [5824814b68901229, c6519fb025e64635, fa7996fe5764f060, 02f47ab1d7e7f6b9]
  analysis: [analysis/out/tables/compatibility_matrix.md, analysis/out/tables/backend_attribution.md]
  code_sha: b294050
supersedes: []
superseded_by: []
---

## Claim

On SM89 (L4) with vLLM 0.29.0 and flashinfer 0.6.18, DFlash block drafting with
an `fp8_e4m3` KV cache runs on the FlashInfer backend without crashing and
without measurable loss of acceptance. Hypothesis H1 — that FlashInfer's FP8
attention kernels exist only for SM90, so the configuration is accepted at
config time and fails at kernel dispatch — is refuted on both of its predicted
failure modes.

## Evidence

Backend read from the engine's own startup log, not inferred:

| cell | mech | kv | requested | actual | status | tau |
|---|---|---|---|---|---|---|
| `c6519fb0` | dflash | auto | FLASHINFER | FLASHINFER | ok | 2.716 |
| `5824814b` | dflash | fp8_e4m3 | FLASHINFER | FLASHINFER | ok | **2.697** |
| `fa7996fe` | none | fp8_e4m3 | FLASHINFER | FLASHINFER | ok | – |
| `02f47ab1` | dflash | fp8_e4m3 | TRITON_ATTN | TRITON_ATTN | ok | 2.745 |

## Reasoning

Two controls were established before the test, which is what makes it
interpretable. `fa7996fe` shows FlashInfer serves FP8 KV on the causal path,
and `c6519fb0` shows FlashInfer serves non-causal drafting at BF16 KV. H1
concerns the intersection, so a failure at `5824814b` could not have been
blamed on FlashInfer being unusable or drafting being unsupported.

The second failure mode — "launches but produces wrong output", which the design
calls headline A+ — is ruled out by tau rather than by output comparison. If the
FP8 non-causal kernel returned garbage, the verifier would reject the drafts
built on it and acceptance would collapse toward 1.0. tau is 2.697 against 2.716
for the same backend at BF16, a 0.7% difference.

## What this does NOT establish

- **Not a quality claim.** tau shows the kernel functions; it does not show FP8
  preserves answer quality. The accuracy arm cannot support that (see F008).
- **Not general across hardware.** One GPU (L4). Issue #54690 is a 4090.
- **Not general across versions.** vLLM 0.29.0 / flashinfer 0.6.18 only.
- **Does not identify the fix.** flashinfer#5272, the PR assumed to be the fix,
  is still open, so whatever resolved this is something else and unidentified.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine prefetch --sweep backend
modal run cloud/modal_probe.py --engine vllm --sweep backend
```
