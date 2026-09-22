---
id: F018
title: RETRACTED — the backend axis was never real; --attention-backend does not select
kind: retraction
status: retracted
confidence: high
date: 2026-09-22
evidence:
  runs: [2026-09-21T17-08Z_repeat, 2026-09-21T07-41Z_backend, 2026-09-21T14-5xZ_backend_gaps]
  analysis: [analysis/out/tables/backend_attribution.md]
  code_sha: 95ac8e1
supersedes: [F017, F006]
---

## Claim (withdrawn)

`sweeps/backend.yaml` and `sweeps/backend_gaps.yaml` were built on the
assumption that `--attention-backend X` makes the engine run backend X. It does
not. vLLM 0.29.0 accepts the flag, records it, validates against it — and then
**selects a different backend anyway**. Every cell that *succeeded* ran
whichever backend `auto` would have chosen for its KV dtype.

There was therefore never a backend axis. The 2x4 matrix in F017 has one
backend repeated down its "ok" column.

## Evidence

The flag reaches the server. From `results/logs/154965a1bfbfe3af.log`:

```
non-default args: {... 'attention_backend': 'TRITON_ATTN' ...}
Using FLASHINFER attention backend out of potential backends:
  ['FLASHINFER', 'TRITON_ATTN']
```

Across every bf16 cell, taking the backend from each log's own selection line:

| requested | KV dtype | actually selected |
|---|---|---|
| FLASHINFER | bf16 | FLASH_ATTN |
| TRITON_ATTN | bf16 | FLASH_ATTN |
| FLEX_ATTENTION | bf16 | FLASH_ATTN |
| FLASH_ATTN | bf16 | FLASH_ATTN |
| FLASHINFER | fp8_e4m3 | FLASHINFER |
| TRITON_ATTN | fp8_e4m3 | FLASHINFER |

The candidate list in the log tracks the **KV dtype**, not the flag: four
backends at bf16, two at fp8_e4m3 (matching F017's capability result). The
engine then takes its own preference from that list.

## What survives, and what does not

**Survives — the rejections.** The flag *is* honoured during validation, so
`FLASH_ATTN + fp8_e4m3` and `FLEX_ATTENTION + fp8_e4m3` were genuinely
rejected, by name, with explicit reasons. F017's capability claim for those two
backends stands, as does the `'FP8 KV cache requires FA3 on SM90 or FA4 on
SM100'` message.

**Survives — F002.** The fp8 cells genuinely ran FlashInfer, confirmed by their
selection lines, so "FlashInfer serves FP8 KV with non-causal drafting on SM89"
is unaffected.

**Survives — F003.** That finding was always read from `auto` cells' logs and
is confirmed rather than undermined: bf16 selects FLASH_ATTN, fp8 selects
FLASHINFER.

**Withdrawn — "TRITON_ATTN supports FP8 KV" (F017).** Never tested. That cell
ran FlashInfer.

**Withdrawn — "backend choice perturbs tau more than FP8 does" (F006).** Those
cells all ran the same backend, so the 2.651-2.769 spread cannot be attributed
to backend. Its actual source is the prompt-set change (F013): within a fixed
prompt set the same cell repeats at 2.710/2.710/2.708 and 2.769/2.769/2.769,
spread <= 0.002.

**Mislabelled — F004's per-backend capacity rows.** The FP8/BF16 *ratio* is
computed within one configuration and is unaffected, but the rows should not be
read as comparing backends.

## What caught it

Running the same cells three times and inspecting each log's selection line.
Backend attribution had been verified from logs once, early (F011), and then
trusted as the axis grew — the check was never re-applied to new cells.

## Consequence

Two fixes, not one. The sweep must use `--attention-config`, which appears to
be the current mechanism. More importantly the harness must **verify** the
selected backend against the requested one after boot and record both, so a
silently ignored flag cannot become an axis again. A configuration knob that
is accepted, echoed back, and then disregarded is exactly the failure a
measurement harness has to be built to catch.
